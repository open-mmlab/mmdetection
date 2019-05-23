.PHONY: mmdetection

mmdetection: requirements.txt docker/Dockerfile
	docker build \
		-t mmdetection:latest . \
		-f docker/Dockerfile

clean:
	rm -rf build/
	docker rmi -f mmdetection:latest
