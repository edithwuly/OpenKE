import openke
from openke.config import Trainer, Tester, Predictor
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/NKX/",
	nbatches = 100,
	threads = 8,
	sampling_mode = "normal",
	bern_flag = 1,
	filter_flag = 1,
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/NKX/", "link")

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200,
	p_norm = 1,
	norm_flag = True)


# define the loss function
model = NegativeSampling(
	model = transe,
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size()
)

# train the model
# trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = True)
# trainer.run()
# transe.save_checkpoint('./checkpoint/transe_nkx.ckpt')

# test the model
# transe.load_checkpoint('./checkpoint/transe_nkx.ckpt')
# tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
# tester.run_link_prediction(type_constrain = False)

# predict
transe.load_checkpoint('./checkpoint/transe_nkx.ckpt')
predictor = Predictor(model = transe, data_loader = test_dataloader, use_gpu = True)
links = predictor.run_link_prediction(type_constrain = False)
fout = open("nkx_transe_prediction.txt", "w", encoding="utf-8")
entity2id = open("./benchmarks/NKX/entity2id.txt", encoding="utf-8")
entityId = {}
for line in entity2id.readlines():
	line = line.strip("\n").split("\t")
	if len(line) != 2:
		continue
	entityId[line[1]] = line[0]
entity2id.close()
relation2id = open("./benchmarks/NKX/relation2id.txt", encoding="utf-8")
relationId = {}
for line in relation2id.readlines():
	line = line.strip("\n").split("\t")
	if len(line) != 2:
		continue
	relationId[line[1]] = line[0]
relation2id.close()
fout.write(str(len(links)) + "\n")
for link in links:
	link = link.split(" ")
	head = entityId[link[0]]
	tail = entityId[link[1]]
	relation = relationId[link[2]]
	fout.write(head + "," + relation + "," + tail + "\n")
fout.close()
