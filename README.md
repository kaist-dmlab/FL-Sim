산공과 서버는 python 2.7 을 사용하는데 ns3 는 python3 로 실행해야 됨

DGX 서버 접속
	143.248.84.89
	prosopher / asdfqwer1234

jupyter 설치
	python3 -m pip install --user jupyterlab

Tensorflow 설치
	python3 -m pip install --user tensorflow-gpu==1.14.0

Tensorflow 테스트
	python3 -c "import tensorflow as tf ; print(tf.__version__)"

ns-3 설치 (네트워크 시뮬레이션)
	unzip ns-3-allinone.zip
	cd ns-3-allinone
	./download.py
	./build.py --enable-examples --enable-tests -- --python=/usr/bin/python3.5
	cd ns-3-dev
	./waf shell

ns-3 테스트
	python3 -c "import ns.core"

networkx 설치(그래프 구조체)
	python3 -m pip install --user networkx

tcpdump 설치(tcpdump 명령어 쳤을 때 명령어가 없을 경우)
	cp tcpdump .local/bin/

실행
	~/.local/bin/jupyter lab --allow-root --ip=0.0.0.0 --no-browser