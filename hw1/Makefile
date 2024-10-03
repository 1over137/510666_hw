FXN="sin"
TXT="out.txt"
IN="in.txt"
FMT="jpeg"

.PHONY: plot
plot:
	./trigonometry.py --function=$(FXN)

.PHONY: read
read:
	./trigonometry.py --read=$(IN) --print=$(FMT)

.PHONY: print
print:
	./trigonometry.py --function=$(FXN) --print=$(FMT)

.PHONY: write
write:
	./trigonometry.py --function=$(FXN) --write=$(TXT)