all:
	javac -cp .:jars/* *.java

softmax:
	javac -cp .:jars/* SoftmaxRegression.java

linear:
	javac -cp .:jars/* LinearRegression.java

logistic:
	javac -cp .:jars/* LogisticRegression.java

test:
	javac -cp .:jars/* Test.java

git:
	git add .
	git commit -m "arbitrary commit message"
	git push -u origin master

clean:
	rm *.class
