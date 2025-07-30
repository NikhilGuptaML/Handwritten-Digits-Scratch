from setuptools import find_packages,setup

def get_requirements(path:str)->list[str]:
    '''
    this function will return a list of requirements
    '''
    requirements=[]
    with open("requirement.txt","r") as f:
        requirements=f.readlines()
        requirements=[requirement.replace("\n","") for requirement in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")

        return requirements


setup(
    name="Handwritten Digits Scratch",
    author="Nikhil",
    version="0.0.1",
    author_email="nikhilg.vips@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirement.txt")
)