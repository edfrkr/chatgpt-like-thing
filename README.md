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

<img width="555" height="293" alt="image" src="https://github.com/user-attachments/assets/74fbed03-a94b-44af-bfab-77e6c05a01a8" />

1.3.1. class BigramLanguageModel(nn.Module)

1.3.1.1. def__init__(self, vocab_size)

   self.vocab_size = vocab_size를 통해 모델이 처음 만들어 질 때의 vocabulary size를 불러옵니다. 
   우리가 가진 vocab_size는 27이므로, self.W = nn.Parameter(torch.randn(vocab_size, vocab_size))를 통해서 정규분포로부터 추출한 (27*27) 사이즈의 무작위 수 행렬인 self.W를 만들어냅니다.

1.3.1.2. def forward(self,x)

   x=x.view(-1)입니다. x는 dataloader를 거쳐 나온 데이터이므로, 32*1의 모양입니다. -1을 기준으로 view를 한다는 의미이므로, x는 32개의 숫자의 나열이 됩니다.

   F.one_hot(x,num_classes=self.vocab_size).float()에서는 one hot의 개념을 알아야 합니다. 예를 들어 "."은 0번째이고, "e"는 5번째입니다. 그러므로 "."을 행렬로 표시하면 [1,0,0,...]가 됩니다. "e"를 행렬로 표시하면 [0,0,0,0,1,0,0,...]가 됩니다. 그러므로 "."이나 "e"와 같은 데이터 하나가 들어왔을 때, x_onehot은 1 * 27의 행렬이 되고, self.w는 x_onehot에 곱하는 27 * 27사이즈의 가중치 행렬이 됩니다.

   그러므로 두 행렬곱의 결과인 logits는 1*27의 행렬이 됩니다. 즉, x_onehot에 해당하는 글자 중 "e"라는 글자가 [., a, ..., z]의 27개 문자와 가지는 연관성을 표현하는 행렬이 됩니다. 

<img width="693" height="225" alt="image" src="https://github.com/user-attachments/assets/b7ac8a8b-7f0d-4be5-83ea-1a3b9e8591d8" />

   다만 명시적으로 one_hot을 사용하지 않고, nn.embedding(vocab_size, vocabsize)를 사용해도 같은 연산결과를 얻을 수 있습니다.

1.3.2. model, logits

   model = BigramLanguageModel(vocab_size)를 통해 우리가 사용할 모델의 위에서 정의한 logits = x_onehot @ W 자료임을 정의하며, vocab_size가 인수이므로, W는 27*27임을 알 수 있습니다. 

   logits = model(xb)이므로, 위에서 정의한 xb를 모델에 넣어서 나온 값을 logits에 집어넣습니다. xb는 32 * 1의 모양이므로, xb의 각각의 데이터에 대한 model에서의 결과값이 나올 것입니다. 그러므로 logits는 32*27사이즈의 행렬임을 알 수 있으며, 무작위의 문자 32개에 대해 우리가 가진 vocab의 27개의 문자가 가지는 연관성에 대한 값이 나올 것입니다. 

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

*dataset까지는 notebook1과 내용이 같으므로 모델부터 설명을 시작합니다.

2.1. MLP model

<img width="647" height="345" alt="image" src="https://github.com/user-attachments/assets/1d81307a-3901-4b07-9d8d-8f2eaf35216e" />

   이번에는 직접 self.W을 만들고 x_onehot과 행렬곱하는 방식이 아니라, nn.sequential 함수를 사용합니다. 
   
   먼저 nn.embedding(vocab_size, emb_dim)은 onehot @ self.W를 그대로 진행하는 것과 같습니다. 곧 embedding을 통과한 데이터는 batch_size * block_size * emb_dim 모양의 행렬이며, 각 행은 각각의 데이터가 vocab_size안의 문자들과 가지는 정보를 의미합니다. emb_dim이 10이라는 이야기는, batch_size(64)*block_size(3) 모양의 행렬의 각각의 데이터(글자들)마다 10차원의 정보를 가진다는 의미입니다. 이 점에서 one_hot은 0과 1의 데이터이지만, nn.embedding은 10차원의 연속적 정보라는 점에서 다릅니다. 
   
   이후 nn.flatten을 통해 10차원의 데이터를 가지고 있는 하나의 블록을 이어붙여서, 결론적으로 64 * 30 모양의 데이터로 가공합니다. (batch_size를 64로 지정했기 때문입니다.) 

   nn.linear(block_size * emb_dim, hidden_dim)에서 hidden이란, 문자열에 숨겨져 있는 규칙을 찾는 것입니다. 즉, emm이라는 block에 대해 a가 오는 규칙의 가능성, b가 오는 규칙의 가능성 등등이 있으며, hidden_dim = 200이므로, 200개의 규칙(혹은 특징)을 추출하는 것입니다. 이렇게 하여 "30 * 200" 사이즈의 가중치행렬을 만들고, 이를 통해 "64 * 200" 사이즈의 

   nn.tanh()에서는 지금까지의 처리된 데이터 (64 * 200 모양)에 하이퍼볼릭 탄젠트 값을 씌워 곡선형으로 변환합니다. 

   마지막으로 다시 nn.linear(hidden_dim, vocab_size)를 함으로써 hidden_layer 데이터 (64 * 200)에 200 * 27사이즈의 가중치 데이터를 곱해서 64 * 27사이즈의 최종 logits 데이터가 완성됩니다. 

   즉, nn.sequential()에서 하는 작업은 embedding하고 flatten하여 만든 한 단어의 행렬을, 같은 과정을 통해 나온 다른 단어의 행렬과 linear작업을 하여 단어와 단어 사이의 거리(벡터와 벡터 사이의 거리 또는 각도)가 작은 단어들을 찾아내어 확률을 부여하는 작업입니다. 
   
*training & optimizing과 sampling은 notebook1과 내용이 같으므로, 바로 결과를 설명합니다.

<img width="748" height="18" alt="image" src="https://github.com/user-attachments/assets/ffec0e01-8f4b-48c9-8c85-8bf8bc671da0" />

notebook1보다는 조금 더 그럴싸한 output이 나왔음을 볼 수 있다. 
___

notebook3

<img width="780" height="102" alt="image" src="https://github.com/user-attachments/assets/fb540905-0f4d-4e85-aa09-250889225b9b" />

이번에는 names가 아닌 셰익스피어의 로미오와 줄리엣 작품을 데이터로 사용하며, 전반적인 구조는 notebook2와 같으므로, 필요한 부분만 설명하겠습니다.

3.1 sliding window dataset

3.1.1. class CharSequenceNextCharDataset(Dataset):
<img width="504" height="316" alt="image" src="https://github.com/user-attachments/assets/9e26f86d-3937-4a4b-9ef8-e39551ce5fcf" />

   이 class의 def__init__은 self.dat = data, self.block_size = block_size이므로, 이 클래스를 사용한 dataset의 인수로 넣은 data(text의 모든 글자를 정수화 한 자료)와 block_size(16)을 사용하겠다는 의미입니다. 

   또한 def_len__(self)에서 len(self.data) - self.block_size를 하는 이유는, self.data전체를 x값으로 넣어버리면 남아있는 데이터가 없으므로 정답이 될 문자를 남겨놓을 수 없기 때문입니다.

   def__getitem__에서는 x는 인덱스 위치의 값부터 블록사이즈만큼까지의 데이터를 가지고, y는 인덱스위치에서 블록사이즈 + 1만큼의 위치에 있는 데이터 1개를 의미합니다. 

<img width="602" height="370" alt="image" src="https://github.com/user-attachments/assets/0b5f8020-c6bc-4c9c-9ce3-8e2fa48ecd20" />

   그러므로 xb는 128 * 16, embedding을 통과한 후에는 128 * 16 * 64, fatten을 통과한 후에는 128 * 80, 그다음 linear를 통과하면 128 * 256, tanh를 통과한 후 마지막 linear를 통과하면 128 * 65의 logits가 나올 것입니다. 

*training과 evaluation은 내용이 같으므로 바로 sampling과 결과를 설명합니다.

3.2. sampling 및 결과

<img width="781" height="360" alt="image" src="https://github.com/user-attachments/assets/bd5025c0-0f09-4f24-ad9d-37c2879ea104" />

sampling과정도 notebook2와 크게 다르지는 않습니다. 다만, max_new_tokens의 의미는 start_text로부터 몇 개의 데이터를 추론해낼 것인지에 대한 정보입니다. def에서는 300으로 정의하였으므로 300개의 데이터를 통해 추론해 낼 것이고, print문에서는 400이므로 ROMEO: 뒤에 400개의 데이터를 최종적으로 출력할 것임을 알 수 있습니다. 

<img width="511" height="320" alt="image" src="https://github.com/user-attachments/assets/78cfb113-ae5e-471c-a732-accf74d9ae3e" />

완벽하지는 않지만, 꽤나 영어문장과 같은 모습이 되었음을 볼 수 있습니다. 
___

4. notebook4

*attention이란?

<img width="374" height="452" alt="image" src="https://github.com/user-attachments/assets/f02695b8-1a0d-4e9d-b807-2d3f948aa23c" />

   I am a student.라는 문장을 input한다고 생각을 해보겠습니다. 이때의 input 데이터는 ["I", " ", "a", "m", ..., "."] 입니다. input embedding은 앞선 notebook에서 model을 통해 batchsize*blocksize*emb_dim의 행렬 데이터를 만든것 과 같이 "I", " ", "a", ..., "." 각각에 대해 동시 고차원의 행렬데이터를 만들어주는 것입니다.

   notebook3나 notebook4에서는 1번째~10번째글자와 같이 sliding window dataset이었지만, attention 모델에서는 각각의 글자가 개별로 input됩니다. 그러므로, input되는 글자가 문장에서 몇 번째 index에 있었던 글자인지에 대한 정보를 주어야합니다. 이를 수행하는 것이 positional encoding입니다.

   최종값은 input embedding + position encoding 형태이며, 우리 문장의 "I"를 예시로 들자면 "I"의 의미를 나타내는 행렬 + "첫번째 위치"를 나타내는 행렬로 표시가 될 것입니다. 이 값이 multi-head attention을 통해 input된 데이터의 정보를 담은 행렬이 되고, feed forward에서는 이 행렬의 정보를 추론합니다.

   OUTPUT에는 우리가 한글로 된 데이터를 원해서 "나는 학생입니다."라는 글자를 넣어 output embedding + positional embedding 한다고 하면, 이 대 진행하는 masked multi-head attention은 어떠한 크기의 행렬이 될 것입니다. 그 행렬의 첫 번째 행은 첫 번째 글자인 "나"에 대한 정보이므로 그 뒤의 값은 필요가 없습니다.(이전에 어떤 글자가 나왔는지를 보아야 하므로, 이후에 어떤 글자가 있는지는 필요한 정보가 아니기 때문입니다.) 그러므로 두번째 행은 "는"에 대한 정보이고, "나는" 까지의 정보를 가지고 있습니다. 

   똑같이 이 output은 input에서 나온 값과 같이 multi-head attention을 통하여 두 데이터의 정보를 담은 행렬이 되고, 최종적으로 이 행렬을 linear layer를 통과시켜, softmax를 통해 확률을 도출해 낼 수 있습니다. 즉, input의 글자간의 유사성을 담은 행렬과 output의 글자간의 유사성을 담은 행렬 간의 유사성을 수치화하여, linear를 돌렸을 때 가장 loss가 적은 gradient를 만들어내는 '글자의 정보를 담은 행렬의 각 데이터의 확률'을 만들어내는 것입니다.  

<img width="542" height="60" alt="image" src="https://github.com/user-attachments/assets/9190b04a-e1da-4007-97a5-fc8d1d24232d" />

   attention이 무엇인지 'i am a student'를 기준으로 설명하겠습니다. 우선 쿼리는 input으로 생각할 수 있습니다. 즉 [i, , a, m, a, s, t, u, d, e, n, t]와 같은 데이터라고 생각한다면, key는 query주변의 글자에 대한 데이터라고 생각할  수 있습니다. 즉, "a"라는 query에 대해 " ", "m", "s"와 같은 key가 존재할 수 있습니다. 그리고 각각의 글자는 positonal encoding이 된 상태이므로 위치정보를 가진 글자들입니다. 그러므로 a의 앞에 " "이 오는지, a의 뒤에 " "이 오는지에 따라 query * key의 값은 달라질 것입니다.

   그러므로 Query = [i, , a, m, a, s, t, u, d, e, n, t], 와 Key = [i, , a, m, a, s, t, u, d, e, n, t]를 Key를 transpose하여 행렬곱하면 12 * 12 * A 모양의 글자간의 관계정보가 담긴 행렬이 만들어 질 것입니다. 이를 dimension을 square root한 값으로 나눠주면, 데이터 크기를 기준으로 자료가 표준화되며, 이를 softmax를 통해 각각의 글자에 대한 확률값으로 바꿀 수 있습니다. 이 값(위에서는 편의상 알파벳으로 썻지만, 글자의 의미와 위치값을 가진 숫자로 이루어진 행렬로부터 나온 숫자값)에 진짜 문장인 'i am a student'의 각 글자가 가지고 있는 의미 및 위치정보를 곱하면, 확률적으로 가장 나올법한 알파벳과 그 위치값이 출력됩니다. 

<img width="488" height="274" alt="image" src="https://github.com/user-attachments/assets/49d99c87-d1fc-47b3-833e-d3eef0bb1abf" />

   이 Q@K_transposed 계산을 한 번 하는 것이 single head attention, 이 계산을 고차원으로 진행하는 것이 multi-head attention이라고 이해할 수 있습니다. 

4.1. dataset

<img width="481" height="231" alt="image" src="https://github.com/user-attachments/assets/9b0218f5-432b-42a1-bd36-657abb5afe8c" />

   아까와는 dataset을 정하는 방법이 달라졌습니다. 아까는 x를 1번째부터 10번째까지, y를 11번째 글자로 지정했던 것과는 달리, y는 idx+1 부터 idx+block_size+1로 정의함으로써 2번째부터 11번째 글짜까지로 지정하는 방식이 되었습니다.

4.2. model

<img width="548" height="318" alt="image" src="https://github.com/user-attachments/assets/afe6b80b-76e8-4be1-bf90-54fb43ee4cef" />

   이 클래스에서 attention 모델의 느낌을 볼 수 있습니다. def__init__에서 self.token_embedding과 self.position_embedding을 통해 input embedding과 position embedding을 하고 있음을 알 수 있으며, nn.embedding을 통과한 데이터는 앞선 노트북에서 self.W를 one_hot과 행렬곱한 것과 절차상으로 같은 것입니다. lm_head는 이 과정을 끝낸 후 통과시키는 linear layering임을 볼 수 있습니다.

   또한, def__forward에서 tok+pos를 함으로써 글자의 의미정보와 위치정보를 더하고 있음을 볼 수 있으며, logits에 lm_head로 부터 나온 값을 대입하므로 logits는  linear layering을 거쳐 나온 logits값임을 알 수 있습니다. 

<img width="784" height="271" alt="image" src="https://github.com/user-attachments/assets/316d33e9-e44c-4d72-8023-b2602ff23010" />

   결론적으로, 희곡의 모양새는 하고 있지만 notebook 3보다는 완성도가 낮음을 볼 수 있습니다. 
___

5. notebook 5

   이제는 masked single-head attention을 진행합니다.

5.1. class SingleHeadASelfAttention

<img width="675" height="387" alt="image" src="https://github.com/user-attachments/assets/8dbce597-5404-46d7-b75b-2904c9874f5a" />

   self.key, self.query, self.value를 통해 모델에 기입되는 문장정보로부터 key, query, value 데이터를 만들어내고 있습니다. self.register_buffer는 하삼각행렬, 즉 미래의 글자 정보는 지운 데이터로 만드는 과정입니다. 그리고 def forward를 통해 query @ key_transpose를 계산한 후 softmax를 취하여 확률값인 wei를 계산해내고, 마지막에 wei @ v를 함으로써 가장 확률적으로 나올법한 글자값인 out을 반환해내는 것을 볼 수 있습니다. 

5.2. class TinyattentionLM

<img width="566" height="399" alt="image" src="https://github.com/user-attachments/assets/8a2aa304-7824-4984-a16b-69a0b8abb808" />

   이 class에서는 def__init__에서 input embedding과 position_embedding을 하고, def forward에서 input embedding과 position embedding 값을 더한 값을 정의하고 있습니다. 또한 def_innit에서 5-1에서 정의한 클래스를 사용하고 있으므로, 위의 클래스는 attention의 구조를 만들고, 밑의 클래스는 attention에 들어갈 데이터를 처리하고 attention을 통과시키는 작업이라고 생각할 수 있습니다. 

<img width="847" height="311" alt="image" src="https://github.com/user-attachments/assets/78e32391-2d6a-4057-b968-561bde93cf97" />

   결론적으로 notebook4보다 개선된 모습을 보여주지만 아직 부족함을 볼 수 있습니다.
___

6. notebook6

6.1. multi-head attention

6.1.1. class Head

<img width="736" height="393" alt="image" src="https://github.com/user-attachments/assets/692565a4-3c48-40a9-a74c-6fec007cbcd9" />

notebook 5와 비교했을 때 self.dropoput = nn.Dropout(dropout)이 새롭게 추가된 것을 볼 수 있습니다. dropout의 역할은 말그대로, 데이터 배치를 가지고 연산하는 과정에서 연산 통로를 무작위로 지워버리는 역할입니다. 이 과정에서, 컴퓨터가 빈 칸을 추론하는 과정을 거치게 되고, 이를 통해 앞으로 생길 빈 칸을 추론하는 능력 또한 올라가게 됩니다. 

6.1.2. class multi-head attention

<img width="913" height="283" alt="image" src="https://github.com/user-attachments/assets/f4d6330d-bb8b-4ee5-9848-d596b61a5eae" />

이제는 multihead attention을 구성하는 단계입니다. 이후 나올 class TinyGPT에서 emb_dim을 128로 설정하였고, num_heads를 4로 설정하였으므로, 이는 하나의 head가 32개의 차원을 담당하는 모양입니다. 

이후 proj는 선형대수의 projection으로, 4개의 head가 계산한 값을 projection하여 수직적으로 데이터를 쌓아놓는 과정입니다. 그리고 여기에 dropout을 취하여 프로그램의 정확성을 더 높입니다. 

6.1.3. class FeedForward

<img width="905" height="219" alt="image" src="https://github.com/user-attachments/assets/b33b30c2-4805-41d2-a994-4f9b02366268" />

nn.linear에서 4*embd_dim을 통해 128 * 512 사이즈의 부풀려진 데이터를 만들어냅니다. 그리고 ReLU를 통해 양수값만 남김으로써 multi-head attention으로 만들어낸 가중치가 실제 x의 값을 더 잘 모방하도록 만듭니다. 그리고 다시 순서를 바꾸어 nn.linear를 함으로써 원래의 사이즈로 돌아오고, dropout을 통해 모델의 추론능력을 향상시킵니다. 

6.1.4. class block

<img width="759" height="249" alt="image" src="https://github.com/user-attachments/assets/b8fbd035-d4a5-480c-90c9-5be719afcb1f" />

self.ln1과 self.ln2는 데이터를 정규화시켜주는 역할입니다. 
self.sa에서는 multi-head attention을 통해 글자와 글자 사이의 관계를 만들어내고, self.ffwd에서는 앞서 정의한 feedforward를 통해 추론시스템을 만들어냅니다. 

def forward에서는 우리의 x가 self.ln을통해 정규화되고, 정규화된 데이터가 self.sa를 통과하며 주변 글자와 가지는 관계 정보를 가지게 됩니다. 그 값을 x와 더하므로 x는 기존 x에 주변 글자와 가지는 관계정보가 더해진 값이 됩니다. 그 x에 self.ffwd를 통과한 정규화된 x를 더하므로, x는 기존의 x값과 주변글자와의 관계정보에다가 그 x의 특징이 극대화된 자료라고 할 수 있습니다. 

6.1.5. class TinyGPT

<img width="829" height="416" alt="image" src="https://github.com/user-attachments/assets/94467032-02aa-4ed8-b76f-76132034bfd6" />

이 클래스에서는 input, 즉 "input.txt"로부터 만들어낸 데이터에, 그 각각의 글자가 다른 글자와 가지는 관계정보와 위치정보를 더합니다. 그리고 num_layer는 self.blocks에서 활용되는데 이는 block의 개수가 됩니다. 예를 들어 blocks가 5라고 한다면 첫번째로 글자와 글자를 조합하는 block을 생성하고, 두번째로 조합된 글자와 조합된 글자를 조합하는 block을 생성하며, 세번째로 그렇게 만들어진 block 간의 문법이나 의미 관계를 만들어내고, 네번째로 그러한 의미관계들로 이루어진 block을 통해 문장을 나타내는 block을 만들어내며, 다섯번째로 그러한 문장의 분위기나 말투 등이 고려되어 나타나는 block을 만들어낸다고 이해할 수 있습니다. 

<img width="754" height="327" alt="image" src="https://github.com/user-attachments/assets/34b45c78-f6fa-4668-9b2e-4480a5bf79c7" />

결론적으로 살짝 어색하지만 훌륭한 결과물이 나타나는 것을 볼 수 있습니다.
___

7. Don Quixote

<img width="874" height="121" alt="image" src="https://github.com/user-attachments/assets/e7a2bf39-c4f7-4ebe-a5c5-a64d01a3a016" />

저는 스페인어를 서어서문학과 졸업생 수준으로 할 수 있습니다. 그래서 스페인어 원어로 되어있는 돈키호테 txt 파일을 사용했습니다. 나머지 프로그래밍 구조는 바꾸지 않고, 모든 dropout 비율을 0.2로 설정하여 프로그램의 추론력을 높이고자 했습니다. 

