import torch
import sys
import types
import numpy as np


# Create a temporary module mapping for unpickling
def create_module_mapping():
    """Create a temporary module mapping to handle old checkpoint imports"""
    # Create a temporary util module
    util_module = types.ModuleType('util')
    util_module.vocab = sys.modules[__name__ + '.util.vocab']
    sys.modules['util'] = util_module
    sys.modules['util.vocab'] = sys.modules[__name__ + '.util.vocab']

def get_IITV(pretrained,device='cuda:3'):
    from .model import Improved_ITV
    from .util.vocab import Concept_phrase,Vocabulary

    print(f"loading checkpoint from {pretrained}")
    
    # Create module mapping before loading checkpoint
    create_module_mapping()
    
    
    try:
        checkpoint = torch.load(pretrained,weights_only=False)
    finally:
        # Clean up the temporary module mapping
        if 'util' in sys.modules:
            del sys.modules['util']
        if 'util.vocab' in sys.modules:
            del sys.modules['util.vocab']
    start_epoch = checkpoint['epoch']
    matching_best_rsum = checkpoint['matching_best_rsum']
    classification_best_rsum = checkpoint['classification_best_rsum']

    print("=> loaded checkpoint '{}' (epoch {}, matching_best_rsum {},classification_best_rsum {})"
          .format(pretrained, start_epoch, matching_best_rsum, classification_best_rsum))

    options = checkpoint['opt']
    if not hasattr(options, 'concept_phrase'):
        setattr(options, "concept_phrase", options.concept_phase)
        options.concept_phrase.idx2phrase = options.concept_phrase.idx2phase
        options.concept_phrase.phrase2idx = options.concept_phrase.phase2idx
        options.concept_phrase.phrase2contractphrase = options.concept_phrase.phase2contractphase
        del options.concept_phrase.idx2phase
        del options.concept_phrase.phase2idx
        del options.concept_phrase.phase2contractphase
        options.concept_bank = 'concept_phrase'
    model = Improved_ITV(options)
    
    model.load_state_dict(checkpoint['model'])

    model.vid_encoder.eval()
    model.text_encoder.eval()
    model.unify_decoder.eval()
    
    # Move model to device
    model = model.to(device)
    
    return model


def get_clip_model(device='cuda:3'):
    ##get CLIP B-32 model
    import clip
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    return clip_model, clip_preprocess

def encode_CLIP_text(clip_model, text, device='cuda:3'):
    import clip
    
    # CLIP has a maximum token limit of 77 tokens
    # If text is too long, truncate it
    try:
        # First, try to tokenize to check length
        tokens = clip.tokenize([text], truncate=True).to(device)
    except Exception as e:
        print(f"CLIP tokenization error: {e}")
        # If tokenization fails, try with truncated text
        # CLIP typically handles around 77 tokens, roughly 77*4 characters
        max_chars = 300  # Conservative estimate
        if len(text) > max_chars:
            text = text[:max_chars]
            print(f"Truncating text to {max_chars} characters")
        try:
            tokens = clip.tokenize([text], truncate=True).to(device)
        except Exception as e2:
            print(f"CLIP tokenization still failed after truncation: {e2}")
            return None
    
    try:
        with torch.no_grad():
            feature = clip_model.encode_text(tokens).cpu().numpy()[0]
        return feature
    except Exception as e:
        print(f"CLIP encoding error: {e}")
        return None
    
def encode_BLIP2_text(text,BLIP2_server,BLIP2_feature_file='tmp/BLIP2_feature.npy',device='cuda:3'):
    import socket
    import time
    import subprocess
    import sys
    import os
    
    max_retries = 3
    retry_delay = 1
    
    def restart_blip2_server():
        """Restart the BLIP2 server if it has crashed"""
        try:
            print("🔄 Attempting to restart BLIP2 server...")
            # Remove stale socket if it exists
            if os.path.exists(BLIP2_server):
                os.unlink(BLIP2_server)
            
            # Start server in background with correct conda environment
            process = subprocess.Popen(
                ['conda', 'run', '-n', 'lavis', 'python', 'BLIP2_text_encoder_server.py', '--device', device, '--BLIP2_server', BLIP2_server, '--BLIP2_feature_file', BLIP2_feature_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
                env={'DEVICE': device, 'BLIP2_SERVER': BLIP2_server, 'BLIP2_FEATURE_FILE': BLIP2_feature_file},
                shell=True
            )
            
            print("❌ Failed to restart BLIP2 server")
            return False
            
        except Exception as e:
            print(f"❌ Error restarting BLIP2 server: {e}")
            return False
    
    for attempt in range(max_retries):
        try:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.settimeout(10)  # 10 second timeout
            client.connect(BLIP2_server)
            
            # Send text
            client.send(text.encode('utf-8'))
            
            # Wait for acknowledgment
            ack = client.recv(1024)
            if ack != b'sent':
                raise Exception(f"Server acknowledgment error: {ack}")
            
            # Send completion signal
            client.send(b'done')
            
            # Load the feature
            reply = np.load(BLIP2_feature_file)
            client.close()
            return reply
            
        except (ConnectionRefusedError, ConnectionResetError, BrokenPipeError) as e:
            print(f"❌ BLIP2 server connection error (attempt {attempt + 1}/{max_retries}): {e}")
            
            # Try to restart the server
            if restart_blip2_server():
                print(f"⏳ Retrying with restarted server in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            else:
                if attempt < max_retries - 1:
                    print(f"⏳ Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise Exception("BLIP2 server is not running and could not be restarted. Please start it manually.")
                
        except FileNotFoundError:
            print(f"❌ BLIP2 feature file not found (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                print(f"⏳ Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise Exception("BLIP2 feature file not generated. Server may have crashed.")
                
        except Exception as e:
            print(f"❌ BLIP2 encoding error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"⏳ Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise Exception(f"BLIP2 encoding failed after {max_retries} attempts: {e}")
        finally:
            try:
                client.close()
            except:
                pass

def encode_ImageBind_text(text,ImageBind_server,ImageBind_feature_file='tmp/ImageBind_feature.npy',device='cuda:3'):
    import socket
    import time
    import subprocess
    import sys
    import os
    
    max_retries = 3
    retry_delay = 1
    
    def restart_imagebind_server():
        """Restart the ImageBind server if it has crashed"""
        try:
            print("🔄 Attempting to restart ImageBind server...")
            # Remove stale socket if it exists
            if os.path.exists(ImageBind_server):
                os.unlink(ImageBind_server)
            
            # Start server in background with correct conda environment
            process = subprocess.Popen(
                ['conda', 'run', '-n', 'imagebind', 'python', 'Imagebind_text_encoder_server.py', '--device', device, '--ImageBind_server', ImageBind_server, '--ImageBind_feature_file', ImageBind_feature_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
                env={'DEVICE': device, 'ImageBind_SERVER': ImageBind_server, 'ImageBind_FEATURE_FILE': ImageBind_feature_file},
                shell=True
            )
            
            # Wait for server to start
            for wait_time in range(30):
                time.sleep(1)
                if os.path.exists(ImageBind_server):
                    try:
                        # Test connection
                        test_client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                        test_client.settimeout(2)
                        test_client.connect(ImageBind_server)
                        test_client.close()
                        print("✅ ImageBind server restarted successfully")
                        return True
                    except:
                        continue
            
            print("❌ Failed to restart ImageBind server")
            return False
            
        except Exception as e:
            print(f"❌ Error restarting ImageBind server: {e}")
            return False
    
    for attempt in range(max_retries):
        try:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.settimeout(10)  # 10 second timeout
            client.connect(ImageBind_server)
            
            # Send text
            client.send(text.encode('utf-8'))
            
            # Wait for acknowledgment
            ack = client.recv(1024)
            if ack != b'sent':
                raise Exception(f"Server acknowledgment error: {ack}")
            
            # Send completion signal
            client.send(b'done')
            
            # Load the feature
            reply = np.load(ImageBind_feature_file)
            client.close()
            return reply
            
        except (ConnectionRefusedError, ConnectionResetError, BrokenPipeError) as e:
            print(f"❌ ImageBind server connection error (attempt {attempt + 1}/{max_retries}): {e}")
            
            # Try to restart the server
            if restart_imagebind_server():
                print(f"⏳ Retrying with restarted server in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            else:
                if attempt < max_retries - 1:
                    print(f"⏳ Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise Exception("ImageBind server is not running and could not be restarted. Please start it manually.")
                
        except FileNotFoundError:
            print(f"❌ ImageBind feature file not found (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                print(f"⏳ Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise Exception("ImageBind feature file not generated. Server may have crashed.")
                
        except Exception as e:
            print(f"❌ ImageBind encoding error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"⏳ Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise Exception(f"ImageBind encoding failed after {max_retries} attempts: {e}")
        finally:
            try:
                client.close()
            except:
                pass