from sklearn import tree

def main():
    print("Ball Classification Case study")
    #Load the data
    BallFeartures=[[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[42,1],[110,0],[35,1],[90,0]]

    Labels=["Tennis","Tennis","Cricket","Tennis","Cricket","Tennis","Cricket","Tennis","Tennis","Tennis","Cricket","Tennis","Cricket","Tennis","Cricket"]

    obj = tree.DecisionTreeClassifier()  #Decide the algorithm

    obj = obj.fit(BallFeartures,Labels) #train the model

    print(obj.predict([[36,1],[91,0]])) #Test the model

if __name__=="__main__":
    main()