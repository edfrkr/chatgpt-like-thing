# chatgpt-like-thing

1. Notebook 1 (Bigram Language Model)
<img width="890" height="476" alt="image" src="https://github.com/user-attachments/assets/2d849e7b-1b78-4a61-b1fc-d424dc6020a8" />
1) import torch / import torch.nn as nn / import torch.nn.functional as F
   from troch.utils.data import Dataset, DataLoader / from pathlib import Path / import urllib.request

   해당부분은 코드를 실행하기 위한 library를 불러오는 부분으로,
   import torch는 Pytorch 라이브러리 전체를 import하고, torch.nn은 neural network 즉 수업에 배운 인공지능을 활용한 코딩을 위해 필요한 라이브러리입니다. torch.nn.functional은 torch.nn이 가지고 있는 여러 함수들을 불러오는 명령이며, F.cross_entropy 와 같은 명령을 통해 loss를 계산하는데 쓰입니다.
   Dataset과 DataLoader는 해당 프로젝트에서 사용할 데이터를 보관하고 불러오기 위해 사용합니다. 
   from pathlib import Path는 우리가 사용할 데이터파일이 존재하는지 찾는 명령으로, import urllib.request 명령과 함깨 썼습니다. 이를 통해 "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"의 주소로부터 "names.txt" 파일을 불러오는 명령으로 생각할 수 있습니다.

2) words, chars

  해당부분은 names.txt를 불러온 후, 전처리를 하는 과정입니다. 
  words = open("names.txt", "r").read().splitlines()의 명령은 names.txt를 읽기전용으로 불러온 후, 문자열로 인식하여 줄바꿈을 기준으로 데이터를 분류한 후, 이 데이터를 words로 저장하는 것입니다.
  chars = sorted(list(set("".joins(words))))에서 joins(words)의 데이터를 합치는 명령입니다. "".joins()의 형태이므로 words의 데이터를 공백없이 이어붙이는 것입니다. set("".joins())에서는 "".joins()로 합쳐진 데이터에서 중복값을 제거합니다. 현재 자료의 값은 emma, olivia, ava ... 이므로 set()의 결과는 emaoliv...와 같은 형태가 될 것입니다. 이후 list() 명령을 통해 ["e", "m", "a", "o", ...]와 같은 리스트로 정리가 되며, sorted() 명령을 통해 위의 list는 ["a", "e", "m", "o", ...]와 같이 알파벳 순으 정리가 됩니다. 
  chars = ["."] + chars에서는 위의 chars라는 리스트의 0번째 데이터로 ".", 즉 마침표를 추가하며, 이는 이후 데이터를 구분하는 기준점 역할이 될 것입니다. 

3) stoi, itos, vocab_size, encoded words
   
  이제, chars의 데이터를 컴퓨터가 읽을 수 있도록 하는 과정을 진행합니다.
  names.txt의 모든 이름을 정리했으므로 chars는 [".", "a", "b", ...]의 모양일것입니다. enumerate(chars)는 순서대로 (0,"."), (1,"a"), ...와 같이 알파벳 하나에 숫자 하나를 대응시켜 튜플로 만드는 명령입니다. 이후, 중괄호 {} 안에서 ch:i 로 대응을 하기 때문에 문자를 key로 하고, 숫자를 value로 하는 dictionary를 만드는 과정을 stoi = {ch:i ...}라고 생각하면 됩니다.
  마찬가지로 itos는 우리가 정리한 stoi 딕셔너리에서 데이터를 가져와 숫자를 key로하고 문자를 value로 하는 dictionary를 만들어낸 것이라고 생각할 수 있습니다.
  stoi에는 "." 부터 "z"까지 27개의 값이 있을 것입니다. 이는 나중에 우리가 만든 프로그램이 사용할 수 있는 토큰의 개수를 의미합니다. 
  마지막으로 encoded_words 명령을 보겠습니다. for w in words의 명령을 통해 words의 각각의 데이터를 w라고 표현합니다. 그리고 그 각각의 데이터 w를 구성하는 문자들을 stoi에 대응합니다. 결과적으로 encoded_words는 각각의 w를 구성하는 문자열에 해당하는 숫자로 구성된 list가 됩니다. 우리의 데이터는 "emma", "olivia"... 이므로 [ [2,5,5,1], [16, 13, 10, 22, 10, 2], ...]와 같은 모습의 리스트가 만들어질것입니다. 

  
  
