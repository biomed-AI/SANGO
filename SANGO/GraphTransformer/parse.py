
def parser_add_main_args(parser):
    # dataset
    parser.add_argument('--data_dir', type=str, default='data/scbasset_data/same_organization/mouse_brain_500.h5ad')
    parser.add_argument('--device', type=int, default=3,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--eval_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc', 'f1'],
                        help='evaluation metric')

    # hyper-parameter for model arch and training
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of layers for deep methods')

    # hyper-parameter for GraphTransformer
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--M', type=int,
                        default=30, help='number of random features')
    parser.add_argument('--use_gumbel', action='store_true', help='use gumbel softmax for message passing', default=True)
    parser.add_argument('--use_residual', action='store_true', help='use residual link for each GNN layer', default=True)
    parser.add_argument('--use_bn', action='store_true', help='use layernorm', default=True)
    parser.add_argument('--use_act', action='store_true', help='use non-linearity for each layer')
    parser.add_argument('--use_jk', action='store_true', help='concat the layer-wise results in the final layer')
    parser.add_argument('--K', type=int, default=10, help='num of samples for gumbel softmax sampling')
    parser.add_argument('--tau', type=float, default=0.25, help='temperature for gumbel softmax')
    parser.add_argument('--lamda', type=float, default=1.0, help='weight for edge reg loss')
    parser.add_argument('--rb_order', type=int, default=2, help='order for relational bias, 0 for not use')
    parser.add_argument('--rb_trans', type=str, default='sigmoid', choices=['sigmoid', 'identity'],
                        help='non-linearity for relational bias')
    parser.add_argument('--batch_size', type=int, default=50000)

    parser.add_argument("--train_name_list", type=str, nargs='+', default=["Cerebellum_62216"])
    parser.add_argument("--test_name", type=str, nargs='+', default=["PreFrontalCortex_62216"])
    parser.add_argument("--sample_ratio", type=float, default=0.1)
    parser.add_argument("--edge_ratio", type=float, default=0.0)
    parser.add_argument("--save_path", type=str, default="save")
    parser.add_argument("--save_name", type=str, default="test")




