THETA=45
WIDTH=10
.PHONY: golf
golf: 
	python3 golf.py --plot=$(THETA)

.PHONY: carbon
carbon:
	python3 carbon.py --width=$(WIDTH)