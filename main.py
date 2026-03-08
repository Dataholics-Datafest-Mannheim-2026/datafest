from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

def main():
    print("Hello from datafest!")


if __name__ == "__main__":
    main()
