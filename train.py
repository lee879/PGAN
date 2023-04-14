import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tools import (
    gradient_penalty,
    plot_to_tensorboard,
    save_checkpoint,
    load_checkpoint,
    generate_examples,
)
from progan_net import Discriminator, Generator
from math import log2
from tqdm import tqdm
import config

torch.backends.cudnn.benchmarks = True # 使用cudnn加速

# 开始制作训练数据集
def get_loader(image_size):
    # 使用了pytorch的图像预处理模块
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            #将像素值缩放到[-1, 1]的范围
            transforms.Normalize(
                [0.5 for _ in range(config.CHANNELS_IMG)],
                [0.5 for _ in range(config.CHANNELS_IMG)],
            ),
        ]
    )
    batch_size = config.BATCH_SIZES[int(log2(image_size / 4))]

    #导入数据库
    dataset = datasets.ImageFolder(root=config.DATASET, transform=transform)

    # 创建图片迭代器
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    return loader,dataset


def train_fn(
    critic, # 鉴别器
    gen, # 生成器
    loader, # 数据迭代器
    dataset, # 数据集
    step, # 训练的模型的个数
    alpha, # 图像混合权重系数
    opt_critic, # 鉴别器优化器
    opt_gen, # 生成器优化器
    tensorboard_step, # tensorboard_step
    writer, # tensorboard 数据保存地址
    scaler_gen, # 使用torch混合精度计算F16
    scaler_critic,# 使用torch混合精度计算F16 加快推理速度
):
    loop = tqdm(loader, leave=True)  # tqdm 库来创建一个进度条，用于迭代遍历一个名为loader的对象
    for batch_idx, (real, _) in enumerate(loop): # 返回的是一个批次 ,和一个real_img 的图片数据，没有返回标签，tf需要手动迭代？
        real = real.to(config.DEVICE) # 使用GPU处理
        cur_batch_size = real.shape[0] # 返回的batch的大小 不同的尺寸的图片可能不同 BATCH_SIZES = [32, 32, 32, 16, 16, 16, 16, 8, 4]

        # Train Critic: max E[critic(real)] - E[critic(fake)] <-> min -E[critic(real)] + E[critic(fake)]
        # which is equivalent to minimizing the negative of the expression
        noise = torch.randn(cur_batch_size, config.Z_DIM, 1, 1).to(config.DEVICE) # 使用GPU产生和一个批次大小的噪声

        with torch.cuda.amp.autocast(): # 上下午管理器 表示可以同时使用torth的混合精度计算 ，包括了F16和F32，在兼容计算准确度的同时提高内存的占有率，以及提高运算速度
            fake = gen(noise, alpha, step) # 产生假的图片
            critic_real = critic(real, alpha, step) # real_imag 输入鉴别器得到predict
            critic_fake = critic(fake.detach(), alpha, step) # 剥离fake的一个Tensor ，这个Tensor不会导致鉴别器梯度的更新
            gp = gradient_penalty(critic, real, fake, alpha, step, device=config.DEVICE)
            # 这个是论文中给出的一个损失函数
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + config.LAMBDA_GP * gp
                + (0.001 * torch.mean(critic_real ** 2))
            )

        opt_critic.zero_grad() # 清空判别器中所有参数的梯度，以便在新一轮的反向传播计算中计算新的梯度。
        scaler_critic.scale(loss_critic).backward() # 计算损失函数loss_critic对判别器参数的梯度，并将其存储在相应参数的.grad属性中
        scaler_critic.step(opt_critic) # 将优化器加入到混合计算当中
        scaler_critic.update() # 更新梯度

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        with torch.cuda.amp.autocast(): # 上下午管理器 表示可以同时使用torth的混合精度计算 ，包括了F16和F32，在兼容计算准确度的同时提高内存的占有率，以及提高运算速度
            gen_fake = critic(fake, alpha, step) #
            loss_gen = -torch.mean(gen_fake)

        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        # alpha 不能超过1 ，逐渐加大，但是不能超过1
        alpha += cur_batch_size / (
            (config.PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset) # 论文中有其描述
        )
        alpha = min(alpha, 1) # 将其约束在1里面，取的是alpha,1 思考这里是否可是使用带热重启的余弦退火

        if batch_idx % 50 == 0: # 总共有1w个图片 每个批次32个图片，然后没有100次训练后
            with torch.no_grad():
                fixed_fakes = gen(config.FIXED_NOISE, alpha, step) * 0.5 + 0.5 # 随机给了8张噪声图
            plot_to_tensorboard(
                writer,
                loss_critic.item(),
                loss_gen.item(),
                real.detach(),
                fixed_fakes.detach(),
                tensorboard_step,
            )
            tensorboard_step += 1 # 没n个训练批次迭代一次

        loop.set_postfix( # pytorch的一个进度条也就是每次梯度更新的参数
            gp=gp.item(),
            loss_critic=loss_critic.item(),
        )

    return tensorboard_step,alpha

def main():
    #初始化鉴别器
    gen = Generator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)
    critic = Discriminator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)

    # initialize optimizers and scalers for FP16 training
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))

    scaler_critic = torch.cuda.amp.GradScaler() # 使用torch混合精度计算F16
    scaler_gen = torch.cuda.amp.GradScaler()

    # for tensorboard plotting
    writer = SummaryWriter(r"D:/logs/gan1") # tensorboard 写入文件

    if config.LOAD_MODEL: #如果有本地模型将使用本地模型进行训练
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC, critic, opt_critic, config.LEARNING_RATE,
        )

    gen.train()
    critic.train()
    tensorboard_step = 0 # 初始步长
    # start at step that corresponds to img size that we set in config
    step = int(log2(config.START_TRAIN_AT_IMG_SIZE / 4)) # 从图片的128开始训练
    for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5  # start with very low alpha
        loader, dataset = get_loader(4 * 2 ** step)  # 4->0, 8->1, 16->2, 32->3, 64 -> 4
        print(f"Current image size: {4 * 2 ** step}") # 128尺寸大小开始训练

        #开始第尺寸训练
        for epoch in range(num_epochs): # num_epochs = 4 # 每个批次训练30论
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            tensorboard_step, alpha = train_fn(
                critic,
                gen,
                loader,
                dataset,
                step,
                alpha,
                opt_critic,
                opt_gen,
                tensorboard_step,
                writer,
                scaler_gen,
                scaler_critic,
            )
            if config.SAVE_MODEL:
                save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
                save_checkpoint(critic, opt_critic, filename=config.CHECKPOINT_CRITIC)

        step += 1  # progress to the next img size

if __name__ == "__main__":
    main()