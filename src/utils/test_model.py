

def main(dataset):
    

    ##################################
    # SET UP
    ##################################

    torch.manual_seed(420)
    random.seed(420)
    np.random.seed(420)
    rng = np.random.default_rng(420)

    print(f'torch version is: {torch.__version__}')
    print(f'torchvision version is: {torchvision.__version__}')

    gc.collect()
    torch.cuda.empty_cache()

    DATA_DIR = r'D:\engineering_thesis_data'
    RUNS_DIR = r'D:\__repos\engineering_thesis\src\runs'
    IMG_DIR = r'D:\__repos\engineering_thesis\src\img'

    writer = SummaryWriter(RUNS_DIR)


    ##################################
    # KITTI 2012
    ##################################


    ##################################
    # SCENE FLOW
    ##################################

    flying_things_test = FlyingThings3D(
        data_path=f'{DATA_DIR}/SceneFlow/FlyingThings3D', 
        training=False, 
        transforms=ToTensor()
    )

    scene_flow_test_loader = DataLoader(
        flying_things_test, 
        num_workers=1, 
        batch_size=1, 
        shuffle=True
    )


    ##################################
    # LOAD MODEL
    ##################################

    network = load_model(
        # file_name='stereo_net_2023-01-05___0859__0867.pkl').to('cuda:0')   # KITTI
        file_name='NEW_BEST_stereo_net_2023-01-03___1142__3594.pkl').to('cuda:0')   # SF


    ##################################
    # TEST SCENE FLOW
    ##################################

    total_test_EPE = 0.0
    total_test_3p_error = 0.0
    test_start = datetime.now()
    # timestamp_sf = test_start.strftime("%Y-%m-%d___%H%M__%S%f")[:-4]

    for i, batch in enumerate(scene_flow_test_loader):
        
        image_left = batch[0].to('cuda:0')
        image_right = batch[1].to('cuda:0')
        predicted = network([image_left, image_right])
        groundtruth = batch[2].to('cuda:0')
        # print(groundtruth.shape)
        # print(predicted.shape)

        total_test_EPE += calc_EPE(predicted.squeeze(), groundtruth.squeeze())
        total_test_3p_error += calc_3p_error(predicted, groundtruth)

        # if (i+1) % 100 == 0:
        #     print (f'Test \
        #              Step [{i+1}/{len(scene_flow_test_loader)}], \
        #              Average EPE: {total_test_EPE/(i+1)}, \
        #              Average 3-pixel error: {total_test_3p_error/(i+1)}')

        #     visualize_image(predicted.squeeze(), f'pred_sf_{timestamp_sf}_{i+1}')
        #     visualize_image(groundtruth.squeeze(), f'gt_sf_{timestamp_sf}_{i+1}')

        del image_left
        del image_right
        del groundtruth
        del predicted
        torch.cuda.empty_cache()
        gc.collect()

    print(f'avg test EPE: {total_test_EPE/len(scene_flow_test_loader)}')
    print(f'avg test 3-pixel error: {total_test_3p_error/len(scene_flow_test_loader)}')
    test_finish = datetime.now()
    print(f'testing finished in {test_finish-test_start}')


    ##################################
    # TEST KITTI
    ##################################

    # total_test_EPE = 0.0
    # total_test_3p_error = 0.0
    # test_start = datetime.now()
    # # timestamp_kitti = test_start.strftime("%Y-%m-%d___%H%M__%S%f")[:-4]

    # for i, batch in enumerate(kitti_test_loader):
        
    #     image_left = batch[0].to('cuda:0')
    #     image_right = batch[1].to('cuda:0')
    #     predicted = network([image_left, image_right])
    #     groundtruth = batch[2].to('cuda:0')

    #     not_empty = groundtruth > 0
    #     predicted_flatten = predicted[not_empty]
    #     groundtruth_flatten = groundtruth[not_empty]

    #     total_test_EPE += calc_EPE(predicted_flatten, groundtruth_flatten)
    #     total_test_3p_error += calc_3p_error(predicted, groundtruth)

    #     # if (i+1) % 10 == 0:       
    #     #     visualize_image(predicted_flatten, f'pred_kitti_{timestamp_kitti}_{i+1}')
    #     #     visualize_image(groundtruth_flatten, f'gt_kitti_{timestamp_kitti}_{i+1}')
        
    #     del image_left
    #     del image_right
    #     del groundtruth
    #     del predicted
    #     torch.cuda.empty_cache()
    #     gc.collect()

    # print(f'avg test EPE: {total_test_EPE/len(kitti_test_loader)}')
    # print(f'avg test 3-pixel error: {total_test_3p_error/len(kitti_test_loader)}')
    # test_finish = datetime.now()
    # print(f'testing finished in {test_finish-test_start}')


    ##################################
    # TEAR DOWN
    ##################################

    gc.collect()
    torch.cuda.empty_cache()
    writer.close()


if __name__ == "__main__":
    main('kitti2012')
    # main('kitti2015')
    # main('sceneflow')
