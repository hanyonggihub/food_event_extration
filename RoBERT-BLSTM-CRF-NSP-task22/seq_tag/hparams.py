import os
import argparse
import torch

here = os.path.dirname(os.path.abspath(__file__))
default_pretrained_model_path = os.path.join(here, '../pretrained_models/bert-base-chinese')#Robert-base-chinese
default_train_file = os.path.join(here, '../datasets2/train/')
default_validation_file = os.path.join(here, '../datasets2/dev/')
default_predict_file = os.path.join(here, '../datasets2/predict/')
default_test_file = os.path.join(here, '../datasets2/dev/')
default_output_dir = os.path.join(here, '../saved_models')
default_log_dir = os.path.join(default_output_dir, 'runs')
default_tagset_file = os.path.join(default_output_dir, 'tagset.txt')
default_model_file = os.path.join(default_output_dir, 'model.bin')
default_checkpoint_file = os.path.join(default_output_dir, 'checkpoint.json')
# default_risk_txt_file=os.path.join(here, '../risk_material.txt')
# default_roles_txt_file=os.path.join(here, '../roles.txt')

parser = argparse.ArgumentParser()

parser.add_argument("--pretrained_model_path", type=str, default=default_pretrained_model_path)
parser.add_argument("--train_file", type=str, default=default_train_file)
parser.add_argument("--validation_file", type=str, default=default_validation_file)
parser.add_argument("--output_dir", type=str, default=default_output_dir)
parser.add_argument("--log_dir", type=str, default=default_log_dir)
parser.add_argument("--tagset_file", type=str, default=default_tagset_file)
parser.add_argument("--model_file", type=str, default=default_model_file)
parser.add_argument("--checkpoint_file", type=str, default=default_checkpoint_file)
parser.add_argument("--predict_input_path", type=str, default=default_predict_file)
parser.add_argument("--test_input_path", type=str, default=default_test_file)
# parser.add_argument("--risk_txt_file", type=str, default=default_risk_txt_file)
# parser.add_argument("--roles_txt_file", type=str, default=default_roles_txt_file)

# model
parser.add_argument('--embedding_dim', type=int, default=768, required=False, help='embedding_dim')
parser.add_argument('--rnn_hidden_dim', type=int, default=256, required=False, help='rnn_hidden_dim')
# parser.add_argument('--T_F_Att', type=bool, default=True, required=False, help='T_F_Att')
# parser.add_argument('--pe_dim', type=int, default=40, required=False, help='pe_dim')
# parser.add_argument('--att_size', type=int, default=256, required=False, help='att_size')
parser.add_argument('--rnn_num_layers', type=int, default=1, required=False, help='rnn_num_layers')
parser.add_argument('--rnn_bidirectional', type=bool, default=True, required=False, help='rnn_bidirectional')
parser.add_argument('--TF_token_b', type=bool, default=True, required=False, help='TF_token_b')
parser.add_argument('--TF_ner', type=bool, default=True, required=False, help='TF_ner')
parser.add_argument('--ner_dim', type=int, default=80, required=False, help='ner_dim')
# parser.add_argument('--dropout', type=float, default=0.1, required=False, help='dropout')

#parser.add_argument('--device', type=str, default='cpu')
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")##改改
# parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--max_len", type=int, default=150)
parser.add_argument("--constant_max_len", type=int, default=200)
parser.add_argument("--train_batch_size", type=int, default=8)
parser.add_argument("--validation_batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=40)#3改改30？？
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--weight_decay", type=float, default=0)


hparams = parser.parse_args()
