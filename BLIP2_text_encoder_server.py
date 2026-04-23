from lavis.models import load_model_and_preprocess
import torch
import os
import socket
import time
import numpy as np
import argparse

def main(device='cuda:3',BLIP2_server='tmp/BLIP2.sock',BLIP2_feature_file='tmp/BLIP2_feature.npy'):

    try:
        print(f"🚀 Loading BLIP2 model on device: {device}")
        model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type='pretrain',
                                                                      is_eval=True, device=device)
        print("✅ BLIP2 model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load BLIP2 model: {e}")
        return
    
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if not os.path.exists(os.path.dirname(BLIP2_server)):
        os.mkdir(os.path.dirname(BLIP2_server))
    if os.path.exists(BLIP2_server):
        os.unlink(BLIP2_server)
    server.bind(BLIP2_server)
    server.listen(0)
    print('✅ BLIP2 server ready for connections!')
    print(f"✅ BLIP2 server ready for connections on {BLIP2_server}")
    print(f"✅ BLIP2 server ready for connections on {BLIP2_feature_file}")
    connection_count = 0
    
    while True:
        try:
            connection, address = server.accept()
            connection_count += 1
            query = connection.recv(1024).decode()
            start_time = time.time()
            
            try:
                print(f"📝 Encoding query #{connection_count}: {query}")
                text_input = txt_processors["eval"](query)
                sample = {"text_input": [text_input]}

                with torch.no_grad():
                    features_text = model.extract_features(sample, mode="text")
                    feature = features_text.text_embeds[:, 0, :].cpu().numpy().squeeze()
                    reply = feature.tolist()
                    vec = np.array(reply, dtype=np.float32)
                    np.save(BLIP2_feature_file, vec)
                
                # Send acknowledgment
                connection.send(b'sent')
                
                # Wait for completion signal
                completion_signal = connection.recv(1024)
                if completion_signal == b'done':
                    print(f"✅ BLIP2 encoding #{connection_count} completed successfully")
                else:
                    print(f"⚠️  Unexpected completion signal: {completion_signal}")
                    
            except Exception as e:
                print(f"❌ Error in BLIP2 text encoding #{connection_count}: {e}")
                try:
                    connection.send(b'error')
                except:
                    pass
                
            connection.close()
            end_time = time.time()
            print(f"⏱️  Query #{connection_count} time: {end_time - start_time:.2f} seconds")
            
        except Exception as e:
            print(f"❌ Server connection error: {e}")
            try:
                connection.close()
            except:
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BLIP2 Text Encoder Server')
    parser.add_argument('--device', type=str, default='cuda:3', help='Device to use (e.g., cuda:0, cuda:1)')
    parser.add_argument('--BLIP2_server', type=str, default='tmp/BLIP2.sock', help='BLIP2 server socket file')
    parser.add_argument('--BLIP2_feature_file', type=str, default='tmp/BLIP2_feature.npy', help='BLIP2 feature file')
    args = parser.parse_args()
    
    print(f"🚀 Starting BLIP2 Text Encoder Server on device: {args.device}")
    main(device=args.device,BLIP2_server=args.BLIP2_server,BLIP2_feature_file=args.BLIP2_feature_file)