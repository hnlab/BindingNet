import argparse
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_log",
    help="input log",
    required=True,
)
parser.add_argument(
    "--output_dir",
    help="output directory",
    required=True,
)
args = parser.parse_args()

log_file = args.input_log
with open(log_file, 'r') as f:
    lines = f.readlines()
loss=[]
R2=[]
for line in lines:
    if 'loss' in line:
        loss.append(float(line.split('\t')[1].split(':')[1]))
    if 'Performance on validate set' in line:
        R2.append(float(line.split('\t')[2].split(':')[1]))
x=[i for i in range(len(loss))]
plt.title('loss')
plt.plot(x, loss)
plt.savefig(f'{args.output_dir}/loss.png', dpi=300)
plt.close()

x2=[i for i in range(len(R2))]
plt.title('R2')
plt.plot(x2, R2)
plt.savefig(f'{args.output_dir}/R2.png', dpi=300)
plt.close()
