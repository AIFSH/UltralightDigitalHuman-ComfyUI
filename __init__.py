import os
import sys
import shutil
import folder_paths
import tempfile
import os.path as osp
now_dir = osp.dirname(__file__)
aifsh_dir = osp.join(folder_paths.models_dir,"AIFSH")
udh_ckpt_dir = osp.join(aifsh_dir,"UltralightDigitalHuman")
sys.path.append(now_dir)

import torchaudio
py = sys.executable or 'python'
base_dir = osp.join(now_dir,"udh")
output_dir = folder_paths.get_output_directory()
udh_output_dir = osp.join(output_dir,"UltralightDigitalHuman")
# os.makedirs(udh_output_dir,exist_ok=True)
from huggingface_hub import snapshot_download
hubert_model_dir = osp.join(udh_ckpt_dir,"hubert-large-ls960-ft")

class TrainUltralightDigitalHumanNode:
    def __init__(self,):
        if not osp.exists(osp.join(hubert_model_dir,"pytorch_model.bin")):
            snapshot_download(repo_id="facebook/hubert-large-ls960-ft",
                              local_dir=hubert_model_dir,
                              allow_patterns=["*.bin","*.json"])
        os.environ['hubert_model_dir'] = hubert_model_dir
        
    @classmethod
    def INPUT_TYPES(s):
        # name_list = [name.split('_')[0] for name in os.listdir(udh_output_dir)]
        try:
            name_list = [name.split('@')[0] for name in os.listdir(udh_output_dir)]
        except:
            name_list = []
        return {
            "required":{
                "train_video":("VIDEO",),
                "template_name":("STRING",{
                    "default": "aifsh",
                    "tooltip":f"can be new or in {name_list}",
                }),
                "asr":(['hubert','wenet'],),
                "patient":("INT",{
                    "default":3,
                    "tooltip":"loss decrease in steps,break"
                }),
                "epochs":("INT",{
                    "default":200,
                }),
                "batch_size":("INT",{
                    "default":16,
                }),
                "learning_rate":("FLOAT",{
                    "default":0.001,
                }),
                "train_again":("BOOLEAN",{
                    "default":True,
                }),
                "if_process":("BOOLEAN",{
                    "default":True,
                }),
                "if_syncnet":("BOOLEAN",{
                    "default":True,
                }),
            }
        }
    
    RETURN_TYPES = ("CONFIG",)
    RETURN_NAMES = ("train_result",)

    FUNCTION = "gen_video"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_UltralightDigitalHuman"

    def gen_video(self,train_video,template_name,asr,patient,
                  epochs,batch_size,learning_rate,
                  train_again,if_process,if_syncnet):
        ## Data Process
        template_name = template_name + "@" + asr
        template_output_dir = osp.join(udh_output_dir, template_name)
        dataset_dir = osp.join(template_output_dir,"dataset")
        ckpt_dir = osp.join(template_output_dir,"checkpoint")
        syncnet_ckpt_dir = osp.join(template_output_dir,"syncnet_ckpt")
        ckpt_path = osp.join(ckpt_dir,"min_loss.pth")
        os.environ['ckpt_path'] = ckpt_path
        if not train_again and osp.exists(ckpt_path):
            output_config = {
                "template_name":template_name,
                "asr": asr,
                "dataset_dir": dataset_dir,
                "ckpt_path":ckpt_path
            }
            return (output_config,)
        else:
            os.makedirs(template_output_dir,exist_ok=True)
        os.makedirs(dataset_dir,exist_ok=True)
        os.makedirs(ckpt_dir,exist_ok=True)
        if if_process:
            with tempfile.NamedTemporaryFile(suffix=".mp4",delete=False,dir=dataset_dir) as f:
                if asr == "hubert":
                    cmd = f'ffmpeg -i {train_video} -r 25 {f.name} -y'
                else:
                    cmd = f'ffmpeg -i {train_video} -r 20 {f.name} -y'
                    encoder_path = osp.join(now_dir,"udh/data_utils/encoder.onnx")
                    if not osp.exists(encoder_path):
                        try:
                            shutil.copyfile(osp.join(udh_ckpt_dir,"encoder.onnx"),encoder_path)
                        except:
                            print("to use wenet,Download wenet encoder.onnx and put it in data_utils/")
                print(cmd)
                os.system(cmd)
        
            py_path = osp.join(base_dir,"data_utils","process.py")
            cmd = f"""{py} {py_path} {f.name} --asr {asr}"""
            print(cmd)
            os.system(cmd)
        if if_syncnet:
            py_path = osp.join(base_dir,"syncnet.py")
            cmd = f"""{py} {py_path} --save_dir {syncnet_ckpt_dir} --dataset_dir {dataset_dir} \
                --asr {asr} --patient {patient} --batch_size {batch_size}"""
            print(cmd)
            os.system(cmd)
        
        py_path = osp.join(base_dir,"train.py")
        syncnet_ckpt_path = osp.join(syncnet_ckpt_dir,"min_loss.pth")
        if if_syncnet:
            cmd = f"""{py} {py_path} --save_dir {ckpt_dir} --dataset_dir {dataset_dir} \
                --asr {asr} --use_syncnet --syncnet_checkpoint {syncnet_ckpt_path} \
                    --epochs {epochs} --batchsize {batch_size} --lr {learning_rate} --patient {patient}"""
        else:
            cmd = f"""{py} {py_path} --save_dir {ckpt_dir} --dataset_dir {dataset_dir} \
                --asr {asr} --epochs {epochs} --batchsize {batch_size} --lr {learning_rate} --patient {patient}"""
        print(cmd)
        os.system(cmd)
        output_config = {
            "template_name":template_name,
            "asr": asr,
            "dataset_dir": dataset_dir,
            "ckpt_path":ckpt_path
        }
        return (output_config,)


class InferUltralightDigitalHumanNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "driving_audio":("AUDIO",),
                "train_result":("CONFIG",),
                "offset":("INT",{
                    "default": 10,
                    "tooltip":"set offset to fix bug pred",
                    "max":40,
                    "min":0,
                    "step":10,
                    "display":"slider",
                })
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "gen_video"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_UltralightDigitalHuman"

    def gen_video(self,driving_audio,train_result,offset):
        infer_dir = osp.join(udh_output_dir,train_result['template_name'],"inference")
        os.makedirs(infer_dir,exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".wav",delete=False,dir=infer_dir) as f:
            waveform = driving_audio['waveform'][0]
            sample_rate = driving_audio["sample_rate"]
            torchaudio.save(f.name,waveform,sample_rate)
        asr = train_result['asr']
        audio_base_name = osp.basename(f.name)[:-4]
        if asr == "hubert":
            py_path = osp.join(base_dir,"data_utils/hubert.py")
            cmd = f"""{py} {py_path} --wav {f.name}"""
            audio_feat = osp.join(infer_dir,f"{audio_base_name}_hu.npy")
        else:
            py_path = osp.join(base_dir,"data_utils/wenet_infer.py")
            cmd = f"""{py} {py_path} {f.name}"""
            audio_feat = osp.join(infer_dir,f"{audio_base_name}_wenet.npy")
        print(cmd)
        os.system(cmd)

        save_path = osp.join(infer_dir,f"{audio_base_name}.mp4")
        py_path = osp.join(base_dir,"inference.py")
        cmd = f"""{py} {py_path} --offset {offset} --asr {asr} --dataset {train_result['dataset_dir']} \
            --audio_feat {audio_feat} --checkpoint {train_result['ckpt_path']} --save_path {save_path}"""

        print(cmd)
        os.system(cmd)
        with tempfile.NamedTemporaryFile(suffix=".mp4",delete=False,dir=output_dir) as v:
            cmd = f"""ffmpeg -i {save_path} -i {f.name} -c:v libx264 -c:a aac {v.name} -y"""
            print(cmd)
            os.system(cmd)
        return (v.name, )

NODE_CLASS_MAPPINGS = {
    "TrainUltralightDigitalHumanNode":TrainUltralightDigitalHumanNode,
    "InferUltralightDigitalHumanNode": InferUltralightDigitalHumanNode
}
