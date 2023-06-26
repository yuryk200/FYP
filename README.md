Welcome to Chemsolve, in this repo you can download the android app chemsolve and the machine learning model im2smilesv2.

The App take pictures of Hydrocarbons you want to classify, the hydrocarbon is then sent to the server running the ML model which will send back the information about the hydrocarbon back to the app.

Once the information is retrieved you can get the app to display a 3D rendering of the hydrocarbon you sent to be scanned, process shown in images below.

![Screenshot_20230626-041550_chemsolve2](https://github.com/yuryk200/chemsolve/assets/82842394/ca8d889d-f051-436d-b3ed-960f71b5212c)
![Screenshot_20230626-041538_chemsolve2](https://github.com/yuryk200/chemsolve/assets/82842394/5d675df0-7a67-4d62-8d2b-14c9898b65f2)
![Screenshot_20230626-041748_chemsolve2](https://github.com/yuryk200/chemsolve/assets/82842394/3ae3217a-10ff-4a79-ba58-d8efdd2223cc)

#Note

For this to work you need to be running the server.py script in the folder im2smilesv2, I usually run this in Visual Studio. For the server to run and be fully functional you need to set on the python environment to the file in FYPFinal PythonEnviroment.yaml, otherwise the ML model won’t work.

Also you need an OpenVPN account on your phone and computer and have them both active when using app and server and also in the android app it is important to change the url IP to the IP of the OpenVPN running on the laptop/computer as this can change everytime you turn on OpenVPN, process for this shown below.

Use this IP in OpenVPN
![Picture1](https://github.com/yuryk200/chemsolve/assets/82842394/3d2054a7-1b34-48d9-a995-315feba1eb7c)

In this URL in the chemsolve app
![Picture2](https://github.com/yuryk200/chemsolve/assets/82842394/90ff6d4c-167f-4f5e-8f34-71c3fdb658e9)
