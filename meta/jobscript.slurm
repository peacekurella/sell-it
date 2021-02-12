#!/ bin / bash −l 
#SBATCH --time =24:00:00 
#SBATCH --ntasks=12 
#SBATCH --mem=32g 
#SBATCH --tmp=32g 
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=girdh005@umn.edu
#SBATCH --gres=gpu:v100:1 
#SBATCH -p v100
 
cd /home/jusun/girdh005/sell-it 
export PATH=/home/jusun/girdh005/.conda/envs/sell-it/bin:$PATH 
python train.py

