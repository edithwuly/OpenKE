import openke
from openke.config import Trainer, Tester, Predictor
from openke.module.model import RESCAL
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/ZDX/",
	nbatches = 100,
	threads = 8,
	sampling_mode = "normal",
	bern_flag = 1,
	filter_flag = 1,
	neg_ent = 25,
	neg_rel = 0
)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/ZDX/", "link")

# define the model
rescal = RESCAL(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 50
)

# define the loss function
model = NegativeSampling(
	model = rescal,
	loss = MarginLoss(margin = 1.0),
	batch_size = train_dataloader.get_batch_size(),
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 0.1, use_gpu = True, opt_method = "adagrad")
trainer.run()
rescal.save_checkpoint('./checkpoint/rescal_zdx.ckpt')

# test the model
# rescal.load_checkpoint('./checkpoint/rescal.ckpt')
# tester = Tester(model = rescal, data_loader = test_dataloader, use_gpu = True)
# tester.run_link_prediction(type_constrain = False)

# predict
rescal.load_checkpoint('./checkpoint/rescal_zdx.ckpt')
predictor = Predictor(model = rescal, data_loader = test_dataloader, use_gpu = True)
links = predictor.run_link_prediction(type_constrain = False)
fout = open("zdx_rescal_prediction.txt", "w", encoding="utf-8")
entity2id = open("./benchmarks/ZDX/entity2id.txt", encoding="utf-8")
entityId = {}
for line in entity2id.readlines():
	line = line.strip("\n").split("\t")
	if len(line) != 2:
		continue
	entityId[line[1]] = line[0]
entity2id.close()
relation2id = open("./benchmarks/ZDX/relation2id.txt", encoding="utf-8")
relationId = {}
for line in relation2id.readlines():
	line = line.strip("\n").split("\t")
	if len(line) != 2:
		continue
	relationId[line[1]] = line[0]
relation2id.close()
for link in links:
	link = link.split(" ")
	if len(link) != 3:
		continue
	head = entityId[link[0]]
	tail = entityId[link[1]]
	relation = relationId[link[2]]
	fout.write(head + "," + relation + "," + tail + "\n")
fout.close()