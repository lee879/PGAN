
import torch

START_TRAIN_AT_IMG_SIZE = 4
DATASET = r'D:\sgan\seeprettyface_age_kids\task\progan\datas'
CHECKPOINT_GEN = "generator.pth"
CHECKPOINT_CRITIC = "critic.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = False
LEARNING_RATE = 1e-3
BATCH_SIZES = [32, 16, 8, 4, 4, 4, 2, 2, 1]
CHANNELS_IMG = 3
Z_DIM = 256  # should be 512 in original paper 论文中给出的输入是一个1x1x512的latent ，这里了给的是256是因为硬件的问题
IN_CHANNELS = 256  # should be 512 in original paper
CRITIC_ITERATIONS = 1
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [10] * len(BATCH_SIZES) # 每个图片训练的epoch
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE) #使用gpu参数随机噪声
NUM_WORKERS = 4 #使用图片迭代器的个数（如果使用的太多会出现内存不足）