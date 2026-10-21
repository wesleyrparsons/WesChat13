**Using UDPipeConvertPlus.pas**

1\. Compile Code

fpc UDPipeConvertPlus.pas



2\. Run Code

UDPipeConvertPlus.exe input.conllu output.txt



3\. Example output

*Input CoNLL‑U: Code*

*2   dog   dog   NOUN   NN   Number=Sing|Person=3   3   nsubj*



*Output: Code*

*dog.noun.sg.3p*







**Using UDPipe.exe**

udpipe.exe --tokenize --tag --parse english-ewt-ud-2.4-190531.udpipe input.txt > output.conllu



