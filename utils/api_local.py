from utils.inference import chat_loop
import argparse


def VicunaChatCompletion(question):

    answer = chat_loop(
                '/home/repo2023/FastChat/13B',
                'cuda',#args.device,
                1, #args.num_gpus,
                None, #args.max_gpu_memory,
                False, #args.load_8bit,
                False, #args.cpu_offloading,
                None,#args.conv_template,
                0.7,# args.temperature,
                512, #args.max_new_tokens,
                False, #args.debug,
                question,
            )
    return answer

