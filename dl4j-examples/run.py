import os, sys

arg = sys.argv[1].split("/")[2:]
arg = '.'.join(arg)
arg = arg[:-6]

command = "java -cp ./target/classes:$(ls target/dependency/*.jar | xargs | sed 's/ /:/g') %s"%arg

print(command)
os.system(command)
