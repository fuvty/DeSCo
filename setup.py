from setuptools import setup, find_packages

setup(
    name="desco",
    version="0.1",
    author="tianyu",
    author_email="fuvty@outlook.com",
    description="Towards Scalable Deep Subgraph Counting",

    # 项目主页
    # url="http://iswbm.com/", 

    packages=find_packages(),

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],

    # 希望被打包的文件
    package_data={
        '':['*.py'],
        'subgraph_counting':['*.py']
    },
    
    # 不打包某些文件
    exclude_package_data={
        'data':['*'],
        'ckpt':['*'],
        'results':['*'],
        'test':['*'],
        'analysis':['*'],
        'migrate':['*'],
    },
)