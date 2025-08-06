from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.got10k_path = '/mnt/DATA/datasets/got-10k'
    settings.save_dir = './results/'
    settings.got_packed_results_path = './results/'
    settings.got_reports_path = './results/'
    settings.lasot_path = '/mnt/DATA/datasets/lasot'
    settings.result_plot_path = './results/result_plots/'



    # settings.results_path = './results/tracking_results/GOT-10k' 
    settings.results_path = './results/tracking_results/test2'



    settings.trackingnet_path = '/mnt/DATA/datasets/TrackingNets'
    settings.uav_path ='/home/ardi/Desktop/dataset/UAV123'
    settings.vot18_path = '/home/ardi/Desktop/dataset/vot-st2018'
    settings.vot20_path = ''
    settings.vot22_path = '/home/ardi/Desktop/dataset/vot2022/stb'
    settings.uav123_10fps_path = '/home/ardi/Desktop/dataset/UAV123_10fps'
    # settings.uav20l_path = 'D:/Dataset_UAV123/UAV123'
    settings.nfs_path = 'E:/main_project/Dataset/NFS/NFS'
    settings.otb_path = '/home/ardi/Desktop/dataset/otb100/raw'
    settings.uavdt_path = '/home/ardi/Desktop/dataset/uavdt/sot'
    settings.visdrone_path = '/home/ardi/Desktop/dataset/visdrone/VisDrone2019-SOT-test-dev'
    settings.dtb70_path = '/home/ardi/Desktop/dataset/DTB70'
    settings.uavtrack_path = '/home/ardi/Desktop/dataset/UAVTrack112'
    settings.uavtrackl_path = '/home/ardi/Desktop/dataset/UAVTrack112l'

    settings.uav20l_path = '/home/ardi/Desktop/dataset/UAV123'



    return settings