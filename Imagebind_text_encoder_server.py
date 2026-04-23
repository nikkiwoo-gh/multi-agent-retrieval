import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'ImageBind'))

from ImageBind.models import imagebind_model
from ImageBind.models.imagebind_model import ModalityType
from ImageBind.data import load_and_transform_text
import torch
import os
import socket
import time
import numpy as np
import argparse

def main(device='cuda:3',ImageBind_server='tmp/ImageBind.sock',ImageBind_feature_file='tmp/ImageBind_feature.npy'):
    
    try:
        print(f"🚀 Loading ImageBind model on device: {device}")
        model = imagebind_model.imagebind_huge(pretrained=True)
        model.eval()
        model.to(device)
        print("✅ ImageBind model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load ImageBind model: {e}")
        return
    
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if not os.path.exists(os.path.dirname(ImageBind_server)):
        os.mkdir(os.path.dirname(ImageBind_server))
    if os.path.exists(ImageBind_server):
        os.unlink(ImageBind_server)
    server.bind(ImageBind_server)
    server.listen(0)
    print('✅ ImageBind server ready for connections!')
    print(f"✅ ImageBind server ready for connections on {ImageBind_server}")
    print(f"✅ ImageBind server ready for connections on {ImageBind_feature_file}")
    connection_count = 0
    
    while True:
        try:
            connection, address = server.accept()
            connection_count += 1
            query = connection.recv(1024).decode()
            start_time = time.time()
            
            try:
                print(f"📝 Encoding query #{connection_count}: {query}")
                inputs = {
                    ModalityType.TEXT: load_and_transform_text([query], device),
                }
                with torch.no_grad():
                    embeddings = model(inputs)
                    feature = embeddings[ModalityType.TEXT]
                    feature = feature.data.cpu().numpy()[0]
                    vec = np.array(feature, dtype=np.float32)
                    np.save(ImageBind_feature_file, vec)
                
                # Send acknowledgment
                connection.send(b'sent')
                
                # Wait for completion signal
                completion_signal = connection.recv(1024)
                if completion_signal == b'done':
                    print(f"✅ ImageBind encoding #{connection_count} completed successfully")
                else:
                    print(f"⚠️  Unexpected completion signal: {completion_signal}")
                   
            except Exception as e:
                print(f"❌ Error in ImageBind text encoding #{connection_count}: {e}")
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
    parser = argparse.ArgumentParser(description='ImageBind Text Encoder Server')
    parser.add_argument('--device', type=str, default='cuda:3', help='Device to use (e.g., cuda:0, cuda:1)')
    parser.add_argument('--ImageBind_server', type=str, default='tmp/ImageBind.sock', help='ImageBind server socket file')
    parser.add_argument('--ImageBind_feature_file', type=str, default='tmp/ImageBind_feature.npy', help='ImageBind feature file')
    args = parser.parse_args()
    
    print(f"🚀 Starting ImageBind Text Encoder Server on device: {args.device}")
    main(device=args.device,ImageBind_server=args.ImageBind_server,ImageBind_feature_file=args.ImageBind_feature_file)