def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')
    
    parser.add_argument('--model', type=str, default='AlexNet', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='/home/dengyh/dataset/cifar10',
                        help='data directory')

    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)
    
    parser.add_argument('--model_save_path', type=str, default='../../../fedml_api/model/pretrained/',
                        help='model save path')
    
    parser.add_argument('--resource_constrained',action='store_true',
                    help='if clients are resource constrained', default=False)
    
    # FL settings

    parser.add_argument('--epochs', type=int, default=1, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--client_num_in_total', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')
    
    parser.add_argument('--join_ratio', type=float, default=0.1,
                        help='Ratio for (client each round) / (client num in total)')
    # parser.add_argument('--client_num_per_round', type=int, default=10, metavar='NN',
    #                     help='number of workers')

    parser.add_argument('--comm_round', type=int, default=5,
                        help='how many round of communications we shoud use')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')
    
    parser.add_argument('--agg_method', type=int, default=0, metavar='N',
                        help='how to aggregate pruned parames, 0:HeteroFL, 1:HeteroFL datasize weighted, 2:HeteroFL datasize and loss weighted, 3:HeteroFL distributed, 4:HeteroFL grouped')
    
    parser.add_argument('--similarity_method', type=int, default=0, metavar='S',
                        help='how to compute the data distribution distance, 0:cosine_similarity, 1:l1 norm, 2:l2 norm')
    
    parser.add_argument('--group_num', type=int, default=2,
                        help='group number (used for agg_method=4)')


    # parser.add_argument('--ci', type=int, default=0,
    #                     help='CI')
    
    # data partition settings
    parser.add_argument('--partition_method', type=int, default=0, metavar='P',
                        help='how to partition the dataset on local workers, 0:iid, 1:shard noniid, 2:Dirichlet noniid, 3:k class log-normal, -1:per user')

    parser.add_argument('--partition_alpha', type=float, default=0.9, metavar='PA',
                        help='partition alpha (default: 0.9)')
    
    parser.add_argument('--datasize_per_client', type=int, default=-1,
                        help='the number of data per client (default: Divide the entire data set evenly)')
    
    parser.add_argument('--num_shards_per_user', type=int, default=2,
                        help='used for partition_method=1, the number of shards allocated to each client')
    
    parser.add_argument('--num_classes_per_user', type=int, default=2,
                        help='used for partition_method=3, the number of classes allocated to each client')
    
    parser.add_argument('--sample_num_per_shard', type=int, default=30,
                        help='the number of samples of per shard')
    
    parser.add_argument('--global_dataset_selected_ratio', type=float, default=-1,
                        help='selected a part of global dataset for training and inference')
    
    parser.add_argument("--max_shards_num", type=int)
    parser.add_argument("--min_shards_num", type=int)
    
    return parser