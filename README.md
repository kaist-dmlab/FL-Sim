����� ������ python 2.7 �� ����ϴµ� ns3 �� python3 �� �����ؾ� ��

DGX ���� ����
	143.248.84.89
	prosopher / asdfqwer1234

jupyter ��ġ
	python3 -m pip install --user jupyterlab

Tensorflow ��ġ
	python3 -m pip install --user tensorflow-gpu==1.14.0

Tensorflow �׽�Ʈ
	python3 -c "import tensorflow as tf ; print(tf.__version__)"

ns-3 ��ġ (��Ʈ��ũ �ùķ��̼�)
	unzip ns-3-allinone.zip
	cd ns-3-allinone
	./download.py
	./build.py --enable-examples --enable-tests -- --python=/usr/bin/python3.5
	cd ns-3-dev
	./waf shell

ns-3 �׽�Ʈ
	python3 -c "import ns.core"

networkx ��ġ(�׷��� ����ü)
	python3 -m pip install --user networkx

tcpdump ��ġ(tcpdump ��ɾ� ���� �� ��ɾ ���� ���)
	cp tcpdump .local/bin/

����
	~/.local/bin/jupyter lab --allow-root --ip=0.0.0.0 --no-browser