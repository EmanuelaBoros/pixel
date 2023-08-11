# MasakhaNER
git clone https://github.com/masakhane-io/masakhane-ner.git data/masakhane-ner
  
# UD data for parsing and POS tagging
wget -qO- https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-4758/ud-treebanks-v2.10.tgz | tar xvz -C data
  
# SNLI for robustness experiments
cd data
wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip snli_1.0.zip && rm -rf snli_1.0.zip
cd ..
