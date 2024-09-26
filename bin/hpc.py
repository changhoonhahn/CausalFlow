'''

python script to deploy flow training on a HPC. The script in this case is for
the Della cluster on the Princeton Research Computing 


'''
import os, sys 


def train_flows(treat_or_control, nf_model='maf', study_name='test', output_dir='.', 
        hr=12, gpu=True): 
    ''' write, deploy, and delete script to train flows using optuna. 
    '''
    jname = "flow.%s.%s" % (treat_or_control, nf_model)
    ofile = "o/_flow.%s.%s" % (treat_or_control, nf_model)
    while os.path.isfile(ofile): 
        jname += '_'
        ofile += '_'

    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s" % jname,
        "#SBATCH --nodes=1", 
        "#SBATCH --time=%s:59:59" % str(hr-1).zfill(2),
        "#SBATCH --export=ALL", 
        ['', "#SBATCH --gres=gpu:1"][gpu], 
        "#SBATCH --output=%s" % ofile, 
        "#SBATCH --mail-type=all",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate sbi", 
        "",
        "python flow_optuna.py %s %s %s %s" % (treat_or_control, nf_model, study_name, output_dir), 
        "",
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None


def train_supports(treat_or_control, study_name='test', output_dir='.', hr=12, gpu=True): 
    ''' write, deploy, and delete script to train flows using optuna. 
    '''
    jname = "supp.%s" % (treat_or_control)
    ofile = "o/_supp.%s" % (treat_or_control)
    while os.path.isfile(ofile): 
        jname += '_'
        ofile += '_'

    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s" % jname,
        "#SBATCH --nodes=1", 
        "#SBATCH --time=%s:59:59" % str(hr-1).zfill(2),
        "#SBATCH --export=ALL", 
        ['', "#SBATCH --gres=gpu:1"][gpu], 
        "#SBATCH --output=%s" % ofile, 
        "#SBATCH --mail-type=all",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate sbi", 
        "",
        "python support_optuna.py %s %s %s" % (treat_or_control, study_name, output_dir), 
        "",
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None

#train_flows('treated', nf_model='made', study_name='test.treated', hr=1, gpu=False)
#train_flows('control', nf_model='made', study_name='test.control', hr=1, gpu=False)

train_supports('treated', study_name='test.support.treated', hr=1, gpu=False)
train_supports('control', study_name='test.support.control', hr=1, gpu=False)
