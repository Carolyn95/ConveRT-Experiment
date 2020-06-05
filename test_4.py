"""
# this is for examining sentences with number of tokens > 40
# are they meaningful?

# sentences in every email (one email may contain more than one sentence)

"""
import gluonnlp as nlp
import multiprocessing as mp
import pdb
from rich import print
import time
from mxnet import gluon
MAX_TOKENS = 40


# process small
class JsonProcessor():

  def __init__(self, train_path, valid_path):
    self.tokenizer = nlp.data.SpacyTokenizer('en')
    self.length_clip = nlp.data.ClipSequence(MAX_TOKENS)
    with open(train_path, 'r') as f:
      train = f.read()
      self.train = train.splitlines()
    with open(valid_path, 'r') as f:
      valid = f.read()
      self.valid = valid.splitlines()

  def getText(self):
    # get data into format => ['string', 'string']
    self.train_sents = []
    self.valid_sents = []
    self.train_labels = []
    self.valid_labels = []

    for t in self.train:
      t = eval(t)
      content = t['content']
      if len(content) != 0:
        # content = '.'.join(content)
        self.train_sents.append(content)
        self.train_labels.append(t['intent'])

    for v in self.valid:
      v = eval(v)
      content = v['content']
      if len(content) != 0:
        # content = '.'.join(content)
        self.valid_sents.append(content)
        self.valid_labels.append(v['intent'])


if __name__ == '__main__':
  train_path = 'data/ntuc_train.json'
  valid_path = 'data/ntuc_valid.json'

  jp = JsonProcessor(train_path, valid_path)
  jp.getText()
  for idx, train_sent in enumerate(jp.train_sents):
    for sent in train_sent:
      sent_len = len(jp.tokenizer(sent))
      if sent_len > 32:
        print(idx, ':', sent_len, '=>', sent)
  print('----------')
  for idx, valid_sent in enumerate(jp.valid_sents):
    for sent in valid_sent:
      sent_len = len(jp.tokenizer(sent))
      if sent_len > 32:
        print(idx, ':', sent_len, '=>', sent)
""" # result
61 : 34 => Also, her Left Union Date (LUD) LUD in UCEM is 31 Dec 2019 in UCEM but her membership status is still showing 
as ACTIVE, shouldn’t it be TERMINATED instead
113 : 38 => For UXI related matters, please email to UXI __EMAILADDRESS__ For technical issues on U-CEM, please email to 
NTUC Service Desk __EMAILADDRESS__ Best Regards, Hui Lian
116 : 33 => For system related matters, please contact UXI Team at __EMAILADDRESS__ For technical issues on UCEM, please 
email to NTUC Service Desk __EMAILADDRESS__ and
166 : 34 => Not required to remove my account’s admin access anymore as the original request was due to the success/error 
messages not appearing for staff accounts during testing, this have been resolved
----------
28 : 37 => be also made accessible to Carine Yip (Assistant GS,  __EMAILADDRESS__ Pls advise if any authorisation is 
required before we import the __EMAILADDRESS__ mailbox into her account
83 : 34 => Kevin: For the part that "email plug-in tries to communicate to the server", do you have a log that shows it's 
trying to connect to the server
"""