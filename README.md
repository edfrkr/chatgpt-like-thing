# chatgpt-like-thing

1. Notebook 1 (Bigram Language Model)
1.1.DATA 준비
<img width="890" height="476" alt="image" src="https://github.com/user-attachments/assets/2d849e7b-1b78-4a61-b1fc-d424dc6020a8" />

1.1.1. import torch / import torch.nn as nn / import torch.nn.functional as F
   from troch.utils.data import Dataset, DataLoader / from pathlib import Path / import urllib.request

   해당부분은 코드를 실행하기 위한 library를 불러오는 부분으로, import torch는 Pytorch 라이브러리 전체를 import하고, torch.nn은 neural network 즉 수업에 배운 인공지능을 활용한 코딩을 위해 필요한 라이브러리입니다. torch.nn.functional은 torch.nn이 가지고 있는 여러 함수들을 불러오는 명령이며, F.cross_entropy 와 같은 명령을 통해 loss를 계산하는데 쓰입니다.
   
   Dataset과 DataLoader는 해당 프로젝트에서 사용할 데이터를 보관하고 불러오기 위해 사용합니다. 
   
   from pathlib import Path는 우리가 사용할 데이터파일이 존재하는지 찾는 명령으로, import urllib.request 명령과 함깨 썼습니다. 이를 통해 "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"의 주소로부터 "names.txt" 파일을 불러오는 명령으로 생각할 수 있습니다.

1.1.2. words, chars

  해당부분은 names.txt를 불러온 후, 전처리를 하는 과정입니다. 
  
  words = open("names.txt", "r").read().splitlines()의 명령은 names.txt를 읽기전용으로 불러온 후, 문자열로 인식하여 줄바꿈을 기준으로 데이터를 분류한 후, 이 데이터를 words로 저장하는 것입니다.
  
  chars = sorted(list(set("".joins(words))))에서 joins(words)의 데이터를 합치는 명령입니다. "".joins()의 형태이므로 words의 데이터를 공백없이 이어붙이는 것입니다. set("".joins())에서는 "".joins()로 합쳐진 데이터에서 중복값을 제거합니다. 현재 자료의 값은 emma, olivia, ava ... 이므로 set()의 결과는 emaoliv...와 같은 형태가 될 것입니다. 이후 list() 명령을 통해 ["e", "m", "a", "o", ...]와 같은 리스트로 정리가 되며, sorted() 명령을 통해 위의 list는 ["a", "e", "m", "o", ...]와 같이 알파벳 순으 정리가 됩니다. 
  
  chars = ["."] + chars에서는 위의 chars라는 리스트의 0번째 데이터로 ".", 즉 마침표를 추가하며, [",", "e", "m", "a", "o", ...]와 같은 모양이 됩니다. 이 마침표는 이후 데이터를 구분하는 기준점 역할이 될 것입니다. 

1.1.3. stoi, itos, vocab_size, encoded words
   
  이제, chars의 데이터를 컴퓨터가 읽을 수 있도록 하는 과정을 진행합니다.
  
  names.txt의 모든 이름을 정리했으므로 chars는 [".", "a", "b", ...]의 모양일것입니다. enumerate(chars)는 순서대로 (0,"."), (1,"a"), ...와 같이 알파벳 하나에 숫자 하나를 대응시켜 튜플로 만드는 명령입니다. 이후, 중괄호 {} 안에서 ch:i 로 대응을 하기 때문에 문자를 key로 하고, 숫자를 value로 하는 dictionary를 만드는 과정을 stoi = {ch:i ...}라고 생각하면 됩니다.
  
  마찬가지로 itos는 우리가 정리한 stoi 딕셔너리에서 데이터를 가져와 숫자를 key로하고 문자를 value로 하는 dictionary를 만들어낸 것이라고 생각할 수 있습니다.
  
  stoi에는 "." 부터 "z"까지 27개의 값이 있을 것입니다. 이는 나중에 우리가 만든 프로그램이 사용할 수 있는 토큰의 개수를 의미합니다. 
  
  마지막으로 encoded_words 명령을 보겠습니다. for w in words의 명령을 통해 words의 각각의 데이터를 w라고 표현합니다. 그리고 그 각각의 데이터 w를 구성하는 문자들을 stoi에 대응합니다. 결과적으로 encoded_words는 각각의 w를 구성하는 문자열에 해당하는 숫자로 구성된 list가 됩니다. 우리의 데이터는 "emma", "olivia"... 이므로 [ [5,13,13,1], [15, 12, 9, 22, 9, 1], ...]와 같은 모습의 리스트가 만들어질것입니다. 
  
1.2. Dataset 
<img width="520" height="548" alt="image" src="https://github.com/user-attachments/assets/06e589c8-7e1f-42f5-a8d6-5cad73bf9317" />

1.2.1 class NamesContextDataset(Dataset)

   PyTorch의 Dataset을 상속받는 NamesContextDataset클래스를 만드는 과정입니다.

1.2.1.1. def__init__(self, encoded_words, blocksize)

   NamesContextDataset을 실행하면 def__init__이 호출됩니다. 
   먼저 self.X, self.Y = [], []를 함으로써, NamesContextDataset에 기입된 객체의 x와 y에 해당하는 빈 리스트를 만듭니다. X list는 입력값, Y list는 정답값이 기입될 것입니다.
   
   Encoded_words는 [[5,13,13,1], [15,12,9,22,9,1], [...], ...]와 같은 모양입니다. 그리고, block_size는 한 번에 몇 개의 문자를 읽을지에 대한 정보입니다. 예를 들어 blcok_size가 3이면, emma라는 단어에서 'emm', 'mma'와 같이 세 글자씩 읽는다는 의미입니다. 그러므로 바깥의 for문은 모든 이름의 개수만큼 [0]*block_size 형태의 context라는 리스트를 만드는 것입니다. 이때 [0]을 사용하는 이유는 위에서 이름의 구분점인 "."이라는 마침표에 대응하는 수가 0이기 때문입니다.
   
   안쪽의 for문에서는 word+[0]을 하고있습니다. 예를들어 emma의 경우에는 [5,13,13,1,0]의 모습이 된 것이며, 'emma.'의 모양으로 바꾼 것으로 이해할 수 있습니다. 이 리스트의 각각의 데이터에 대해 X에는 context를 집어 넣습니다. block_size가 3일 때, emma로 예를 들어보겠습니다. emma의 첫번째 ix는 5입니다. 이때, self.X.append(context.copy())로 인해 X = [[0,0,0]]의 모양이 되고, Y는 [5] 모양이 됩니다.
즉 [0,0,0]이 x값으로 들어간 것에 대해 y값을 5로 대응하는 것입니다. 문자로 생각하면 [...]에 대해 [e]를 대응시키는 것입니다. 
   
   이후 context = context[1:] + [ix]로 인해 context의 모양은 [0,0] + [5] = [0,0,5]가 됩니다. 이제 emma의 두번째 ix는 13이므로, X = [[0,0,0], [0,0,5]]가 되고, Y는 [5,13]이 됩니다. context는 [0,5] + [13] = [0,5,13]으로 바뀝니다. 이 모든 과정이 끝나면 
   X = [[0,0,0] , [0,0,5], [0,5,13], [5,13,13], [13,13,1]], Y = [5,13,13,1,0] 이 됩니다. 
다음 이름인 olivia까지 진행하면 X = [[~],[~],[~],[~],[~],[~],[~],[~],[~],[~],[~],[~]], Y=[~,~,~,~,~,~,~,~,~,~,~,~]의 모양이 될 것입니다.

   self.X = torch.tensor(), self.Y = torch.tensor()는 우리가 만든 self.X, self.Y 리스트를 tensor 모양으로 바꾸는 과정입니다.

1.2.1.2. def__len__(self), def__getitem(self,idx)
   
   def__len__(self)는 self.Y의 길이를 반환하는 함수이고, def__getitem__(self, idx)는 self.X, self.Y의 idx번째 인덱스 위치에 있는 값을 반환하는 함수입니다. 

1.2.2. block_size, dataset, loader

   notebook에서는 block_size를 1로 설정하였고, 이는 X가 [[0],[5],[13],[13],[1]], Y가 [5,13,13,1,0]이 된다는 것을 의미합니다. 즉 x로 들어온 [0]에 대해 y값으로 5를 대응시키는 것입니다. 

   dataset으로는 block_size를 1로 하여, 위의 encoded_words를 받아들이는 것입니다. 

   그리고 DataLoader(dataset, batch_size=32, shuffle=True)로 함으로써, 위의 dataset으로부터 무작위로 32개의 데이터를 뽑는 loader 시스템을 만들었습니다. 즉, 'emma, olivia,...'의 이름의 개수가 32033개이므로, 32033개의 이름으로 만든 (self.X, self.Y)에서 32개씩의 정보를 각각 무작위추출한다는 의미입니다. 

1.2.3. x, y, xb, yb

   x, y =dataset[0]이므로, x의 첫번째 값과 y의 첫번째 값인 [0], 5 가 나올 것입니다. 
   
   xb, yb = next(iter(loader))이므로, iter(loader)를 통해 위에서 만든 loader로부터 하나씩 데이터를 꺼낸 후, next()를 통해 다음차례의 데이터를 또 꺼내는 것입니다. 그러므로 xb = [[0,-,-], [13,13,1], ~~~] (32개), yb = [13,5,1,~~~] (32개)라고 생각할 수 있습니다.

1.3. Bigram modelling  *명시적 one-hot
<img width="624" height="282" alt="image" src="https://github.com/user-attachments/assets/e7c249a6-4b1e-40be-b07d-f478d6b2fdc4" />

1.3.1. class BigramLanguageModel(nn.Module)

1.3.1.1. def__init__(self, vocab_size)

   self.vocab_size = vocab_size를 통해 모델이 처음 만들어 질 때의 vocabulary size를 불러옵니다. 
   우리가 가진 vocab_size는 27이므로, self.W = nn.Parameter(torch.randn(vocab_size, vocab_size))를 통해서 정규분포로부터 추출한 (27*27) 사이즈의 무작위 수 행렬인 self.W를 만들어냅니다.

1.3.1.2. def forward(self,x)

   x=x.view(-1)입니다. x는 dataloader를 거쳐 나온 데이터이므로, 32*1의 모양입니다. -1을 기준으로 view를 한다는 의미이므로, x는 32개의 숫자의 나열이 됩니다.

   F.one_hot(x,num_classes=self.vocab_size).float()에서는 one hot의 개념을 알아야 합니다. 예를 들어 "."은 0번째이고, "e"는 5번째입니다. 그러므로 "."을 행렬로 표시하면 [1,0,0,...]가 됩니다. "e"를 행렬로 표시하면 [0,0,0,0,1,0,0,...]가 됩니다. 그러므로 "."이나 "e"와 같은 데이터 하나가 들어왔을 때, x_onehot은 1*27의 행렬이 되고, self.w는 x_onehot에 곱하는 27*27사이즈의 가중치 행렬이 됩니다.

   그러므로 두 행렬곱의 결과인 logits는 1*27의 행렬이 됩니다. 즉, x_onehot에 해당하는 글자 중 "e"라는 글자가 [., a, ..., z]의 27개 문자와 가지는 연관성을 표현하는 행렬이 됩니다. 

<img width="693" height="225" alt="image" src="https://github.com/user-attachments/assets/b7ac8a8b-7f0d-4be5-83ea-1a3b9e8591d8" />

   다만 명시적으로 one_hot을 사용하지 않고, nn.embedding(vocab_size, vocabsize)를 사용해도 같은 연산결과를 얻을 수 있습니다.

1.3.2. model, logits

   model = BigramLanguageModel(vocab_size)를 통해 우리가 사용할 모델의 위에서 정의한 logits = x_onehot @ W 자료임을 정의하며, vocab_size가 인수이므로, W는 27*27임을 알 수 있습니다. 

   logits = model(xb)이므로, 위에서 정의한 xb를 모델에 넣어서 나온 값을 logits에 집어넣습니다. xb는 32*1의 모양이므로, xb의 각각의 데이터에 대한 model에서의 결과값이 나올 것입니다. 그러므로 logits는 32*27사이즈의 행렬임을 알 수 있으며, 무작위의 문자 32개에 대해 우리가 가진 vocab의 27개의 문자가 가지는 연관성에 대한 값이 나올 것입니다. 

<img width="471" height="17" alt="image" src="https://github.com/user-attachments/assets/a6ca1c8d-90f6-4b22-b83c-9ed3b97ce900" />
<img width="247" height="16" alt="image" src="https://github.com/user-attachments/assets/0f35e4dd-54dd-4af2-b395-f9349274f6eb" />

   이 logits에 softmax를 통해 확률값을 도출하는 cross_entropy를 진행했을 때, loss값이 3.84정도로 아직은 모델의 정확성이 좋지는 않습니다. 
   
1.4. Training & Evaluating
<img width="440" height="467" alt="image" src="https://github.com/user-attachments/assets/0dab6344-8556-4eb5-b6ca-f6634201116c" />

1.4.1. def train_one_epoch 

   이제 이 모델을 학습합니다. loss = F.cross_entropy(logits, yb)를 함으로써 우리가 x를 통해 도출한 logits의 확률값과 실제 문자값인 yb를 비교한 loss 값이 나옵니다.

   그리고 optimizer.zero_grad, loss.backward, optimizer.step 과정을 거치며 X@W의 기울기룰 지우고, self.W를 고치는 작업을 반복합니다. 

1.4.2. def evaluate

   이번에는 model.eval()을 사용합니다. 즉, training이 아닌 점검에 해당하기에, 학습에 필요한 zero_grad(), backward(), step()이 존재하지 않습니다.
<img width="738" height="111" alt="image" src="https://github.com/user-attachments/assets/3fa5513f-2db7-4f7e-a285-b6c329ba4b8a" />
<img width="347" height="84" alt="image" src="https://github.com/user-attachments/assets/63d2c7ad-d9dc-4eec-8a2f-2627e05f13d7" />

1.5. Sampling
<img width="659" height="417" alt="image" src="https://github.com/user-attachments/assets/d62d8140-f054-4516-91be-8ff221be2143" />

1.5.1. def sample

1.5.1.1. 바깥 for문

   context = torch.zeros를 합니다. 즉, 이전의 코딩에서 만든 context의 내용을 다 지워서 초기 상태로 만드는 것입니다. 

1.5.1.2. 안쪽 for문

   context를 모델에 넣어 나온 logits가 가지고 있는 각각의 숫자(글자)에 대해 softmax를 통해 확률값을 정하고, ix = torch.multinomial을 통해 확률적으로 숫자(글자)를 뽑습니다. 이렇게 뽑힌 숫자를 next_token으로 정하고, 그 숫자에 해당하는 글자를 itos를 통해 구하여 바깥 포문의 out list에 집어넣습니다. next_token == 0이면 "."이 나온 것이므로 이를 중단합니다. 

   그리고 context = torch.cat([context[:, 1:], ix], dim=1)를 통해, 기존의 context의 첫 값을 버리고 next_token에 해당하는 값을 넣어서 새로운 context를 만들고 다시 for문이 진행됩니다. 그러므로 새로 진행되는 for문은 방금 전에 진행한 for문으로 나온 문자 뒤의 나올 문자를 구하는 for문이 됩니다. max_len=20이므로, 안쪽 for문을 통해 나온 결과값의 길이는 최대 20입니다. 이 과정을 바깥 for문이 끝날때까지 하므로, num_samples = 10이므로 10개의 결과값이 나올 것입니다. 
<img width="708" height="20" alt="image" src="https://github.com/user-attachments/assets/48881ea2-da63-4bdb-9ae0-5aef8b2a9a87" />

결과값은 아직 신뢰할만하지 못하는데, 그 이유는 block_size의 문제일 수도 있고, 중복되는 문자열에 대한 고려가 미약해서일 수도 있습니다. 
___

2. notebook 2




   




   
