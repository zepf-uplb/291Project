from hashlib import md5, sha1

def main():
	#hasher = md5()
	hasher = sha1()

	with open("words.txt", "r") as f:
		content = f.readlines()

	content = [x.rstrip() for x in content]		

	'''with open("md5.txt", "w") as f:
		for x in content:
			hasher.update(x.encode())			
			f.write(hasher.hexdigest()+"\n")'''

	with open("sha1.txt", "w") as f:
		for x in content:
			hasher.update(x.encode())			
			f.write(hasher.hexdigest()+"\n")


if __name__ == '__main__':
	main()