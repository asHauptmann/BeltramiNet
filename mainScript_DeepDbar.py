# calling script for Unet_CMR to set options

import DeepDbar as Dbar
import matplotlib.pyplot as plt
import h5py
import numpy as np

trainFL = False
evalFL  = True

if(trainFL):
    
    trainSet = 'placeholder' #Define dataset for training
    testSet  = 'placeholder' #Define dataset for testing during training
    
    dataSet = Dbar.read_data_sets(trainSet,testSet)
    
    bSize               = int(8)
    print("bSize = %d" % bSize)
    
    trainIter           = 100000            # DeepDbar typically was 200,000
    checkPointInterval  = 5000              # In which intervals should we save the network parameter
    useTensorboard      = True              # Use Tensorboard for tracking (need to setup tensorboard and change path)
    filePath            = 'netData/'        # Where to save the network, can be absolute or relative
    
    #####
    
    lossFunc    = 'l2_loss'        #or 'l1_loss'
    unetType    = 'resUnet'        #or 'Unet'
    
    experimentName = 'DeepDbar_' + unetType + '_' + lossFunc + '_test'  #Name for this experiment
    
    Dbar.training(dataSet,bSize,trainIter,checkPointInterval,lossFunc,unetType,useTensorboard,experimentName,filePath)
    
elif(evalFL): 
    
    ## Define network name #only 4max_res supported at the moment
    unetPath= 'netData/BeltramiNet_test_4max_res_Sept.ckpt' 
    #
    networkType = "resUnet"
    
    iii=int(12) 
    # Choose dataset to evaluate
    dataSet  = 'KIT4_recons/recon_R5_c' + str(iii) + 'chest.mat'       #Simulated test data  
#    dataSet  = 'KIT4_recons/recon_R5_c' + str(iii) + 'circ.mat'       #Simulated test data  
    fileOutNameBatch = 'DeepDbar_recon.h5'
    IMAGE_NAME = 'reconRect'
    
    #Load data from matlab
    test_images = Dbar.extract_images(dataSet,IMAGE_NAME)    
    test_input=test_images
    
    # Call Unet
    output_images = Dbar.reconstruction(unetPath,test_input,networkType)
    
    # Save reconstruction if you want to use in MATLAB
    fData = h5py.File(fileOutNameBatch,'w')
    fData['result']= np.array(output_images)        
    fData['imag']= np.array(test_input)
    fData.close()
    
    # Some plot of reconstructed images
    plt.figure(1)
    plt.imshow(output_images[0,:,:])
    plt.figure(2)
    plt.imshow(test_input[0,:,:,0])
    plt.pause(1)
    
    



