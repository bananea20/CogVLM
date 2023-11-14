# -*- encoding: utf-8 -*-
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from tqdm import tqdm
import torch
import time
import argparse
from translate import Translator
import pickle
from sat.model.mixins import CachedAutoregressiveMixin

from utils.chat import chat
from models.cogvlm_model import CogVLMModel
from utils.language import llama2_tokenizer, llama2_text_processor_inference
from utils.vision import get_image_processor

def _check_image_file(path):
    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'pdf'}
    return any([path.lower().endswith(e) for e in img_end])


def get_image_file_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    if os.path.isfile(img_file) and _check_image_file(img_file):
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if os.path.isfile(file_path) and _check_image_file(file_path):
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    parser.add_argument("--top_k", type=int, default=1, help='top k for top k sampling')
    parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
    parser.add_argument("--english", action='store_true', help='only output English')
    parser.add_argument("--version", type=str, default="chat", help='version to interact with')
    parser.add_argument("--from_pretrained", type=str, default="cogvlm-chat", help='pretrained ckpt')
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    parser.add_argument("--no_prompt", action='store_true', help='Sometimes there is no prompt in stage 1')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    parser = CogVLMModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # load model
    model, model_args = CogVLMModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
        deepspeed=None,
        local_rank=rank,
        rank=rank,
        world_size=world_size,
        model_parallel_size=world_size,
        mode='inference',
        skip_init=True,
        use_gpu_initialization=True if torch.cuda.is_available() else False,
        device='cuda',
        **vars(args)
    ), overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {})
    model = model.eval()
    from sat.mpu import get_model_parallel_world_size
    assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=args.version)
    image_processor = get_image_processor(model_args.eva_args["image_size"][0])

    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

    text_processor_infer = llama2_text_processor_inference(tokenizer, args.max_length, model.image_length)

    if not args.english:
        if rank == 0:
            print('欢迎使用 CogVLM-CLI ，输入图像URL或本地路径读图，继续输入内容对话，clear 重新开始，stop 终止程序')
    else:
        if rank == 0:
            print('Welcome to CogVLM-CLI. Enter an image URL or local file path to load an image. Continue inputting text to engage in a conversation. Type "clear" to start over, or "stop" to end the program.')
    
    img_root = '/data/home/bofu.huang.o/xhs_data/new_img/'
    trains = Translator(from_lang="Chinese",to_lang="English")
    word_dict = {}
    


    with torch.no_grad():
        
        all_folders = os.listdir(img_root)
        query_list = [] 
        for i, folder in enumerate(all_folders):
            try:
                en_name = trains.translate(folder)
                word_dict[folder] = en_name
                print(en_name)
            except:
                en_name = folder
                word_dict[folder] = ''
                print(f"无法翻译{folder}")
            tmp = []
            tmp.append(
                    f"Is this image related to the keyword '{en_name}'?  just answer yes or no,then STOP do not answer one more word."
                )
            query_list.append(tmp)
        print(query_list)
        with open('dict_file.pkl', 'wb') as f:  
            pickle.dump(word_dict, f)
            

                
        for i, folder in enumerate(all_folders):
            # if i == 1:
            #     break

            # query_list = [
            #     f'{folder}在图片中吗, 只需回答yes或no, 不要返回其他无关回答',
            #     'Extract all objects in the image',
            #     'Watch is in this image, just output the content'
            # ]
            # query_list = [] 
            # try:
            #     en_name = trains.translate(folder)
            #     if rank == 0:
            #         print(f'翻译{folder}为{en_name}')
            #     # query_list.append(
            #     #     f'Is there a {en_name} in the image, just answer yes or no, no need to answer other words',
            #     # )
            #     query_list.append(
            #         f"Is this image related to the keyword '{en_name}'?  just answer yes or no,then STOP do not answer one more word."
            #     )
            #     # query_list.append(
            #     #     f"why?"
            #     # )
            # except:
            #     if rank == 0:
            #         print(f'翻译{folder}失败')

            img_folder = os.path.join(img_root, folder)
            imgs_in_folder = get_image_file_list(img_folder)
            for j, image_path in enumerate(imgs_in_folder):
                if rank == 0:
                    print(f'{time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))} [{i}/{len(all_folders)}] [{j}/{len(imgs_in_folder)}]')
                # if os.path.exists(image_path+'.json'):
                #     continue
                # print(image_path.replace('.jpg', '') +'.json')
                if os.path.exists(image_path.replace('.jpg', '') +'.json'):
                    continue
                if rank == 0:
                    f_w = open(image_path.replace('.jpg', '') +'.json','w', encoding='utf-8')
                # print('image:',image_path)
                if rank == 0:
                    image_path = [image_path]
                else:
                    image_path = [None]
                history = None
                cache_image = None
                if world_size > 1:
                    torch.distributed.broadcast_object_list(image_path, 0)
                
                image_path = image_path[0]
                assert image_path is not None
                
                res_dict = {}
                
                for query in query_list[i]:
                    # print('query:', query)
                    if rank == 0:
                        query = [query]
                    else:
                        query = [None]
                    if world_size > 1:
                        torch.distributed.broadcast_object_list(query, 0)
                    query = query[0]
                    assert query is not None
                    try:
                        response, history, cache_image = chat(
                            image_path, 
                            model, 
                            text_processor_infer,
                            image_processor,
                            query, 
                            history=history, 
                            image=cache_image, 
                            max_length=args.max_length, 
                            top_p=args.top_p, 
                            temperature=args.temperature,
                            top_k=args.top_k,
                            invalid_slices=text_processor_infer.invalid_slices,
                            no_prompt=args.no_prompt
                            )
                    except Exception as e:
                        print(e)
                        break
                    if rank == 0:
                        res_dict[query] = response
                        # if not args.english:
                        #     print("模型："+response)
                        #     if tokenizer.signal_type == "grounding":
                        #         print("Grounding 结果已保存至 ./output.png")
                        # else:
                        #     print("Model: "+response)
                        #     if tokenizer.signal_type == "grounding":
                        #         print("Grounding result is saved at ./output.png")
                if rank == 0:
                    f_w.write(json.dumps(res_dict))
                    f_w.close()


if __name__ == "__main__":
    main()
