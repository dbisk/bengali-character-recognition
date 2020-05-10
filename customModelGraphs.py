import matplotlib.pyplot as plt
losses = []
accuracies = []
testAccuracies = []
losses.append(1.6145114905090743         )
accuracies.append(0.13136078470424217    )
testAccuracies.append(0.05008962358095997)
losses.append(0.6504757926313111         )
accuracies.append(0.5229224756024696     )
losses.append(0.4357845015921758         )
accuracies.append(0.6625174268074089     )
losses.append(0.34962176277859663        )
accuracies.append(0.7231378211511651     )
losses.append(0.2997243386169963         )
accuracies.append(0.7593855805616411     )
losses.append(0.2636027199155731         )
accuracies.append(0.785040330611432      )
losses.append(0.23708928515003383        )
accuracies.append(0.803550089623581      )
losses.append(0.21494319276273702        )
accuracies.append(0.8209955686118303     )
losses.append(0.19666970585005922        )
accuracies.append(0.8338914060944035     )
losses.append(0.1799700075218232         )
accuracies.append(0.8469428400716988     )
losses.append(0.16639172461155352        )
accuracies.append(0.8557993925512846     )
testAccuracies.append(0.7721071499701255 )
losses.append(0.15531228142536432        )
accuracies.append(0.8645314678350926     )
losses.append(0.14349145388096415        )
accuracies.append(0.8720498904600678     )
losses.append(0.13443601806991157        )
accuracies.append(0.8796305516829317     )
losses.append(0.12438062924115793        )
accuracies.append(0.8868689006174069     )
losses.append(0.11720762534377846        )
accuracies.append(0.892296106353316      )
losses.append(0.10905536640270778        )
accuracies.append(0.8995344552877913     )
losses.append(0.10213025332043153        )
accuracies.append(0.9045571101374228     )
losses.append(0.09688083699536082        )
accuracies.append(0.9083474407488548     )
losses.append(0.08949582251133742        )
accuracies.append(0.9143472415853415     )
losses.append(0.04222564024654892        )
accuracies.append(0.9631111830312686     )
testAccuracies.append(0.8231179047998407 )
losses.append(0.03189429547248259        )
accuracies.append(0.9747746962756423     )
losses.append(0.02859755343262793        )
accuracies.append(0.9778181637124079     )
losses.append(0.025970361094157037       )
accuracies.append(0.9810732423819957     )
losses.append(0.024280406981139147       )
accuracies.append(0.9822619996016729     )
losses.append(0.022547193730475507       )
accuracies.append(0.9845772754431388     )
losses.append(0.02117817642108609        )
accuracies.append(0.9856228838876718     )
losses.append(0.020354476190524117       )
accuracies.append(0.986214150567616      )
losses.append(0.019079888070837457       )
accuracies.append(0.9872659828719379     )
losses.append(0.018317181788190563       )
accuracies.append(0.9884236207926708     )
losses.append(0.017521656086714365       )
accuracies.append(0.9890833499302928     )
testAccuracies.append(0.8241884086835292 )
losses.append(0.016445996743181444       )
accuracies.append(0.9900107050388369     )
losses.append(0.015740799208038884       )
accuracies.append(0.99072022505477       )
losses.append(0.015017882558667001       )
accuracies.append(0.9915915654252141     )
losses.append(0.014536488177160217       )
accuracies.append(0.9918031766580363     )
losses.append(0.013967328058512698       )
accuracies.append(0.9921392650866361     )
losses.append(0.013338412522365777       )
accuracies.append(0.9927740987851026     )
losses.append(0.012681221204593982       )
accuracies.append(0.9932222166899024     )
losses.append(0.01216424140759845        )
accuracies.append(0.99379481179048       )
losses.append(0.011827388734933936       )
accuracies.append(0.99396285600478       )
losses.append(0.010723410385198319       )
accuracies.append(0.9950707030472018     )
testAccuracies.append(0.8225453096992631 )
losses.append(0.010462801379802752       )
accuracies.append(0.9954254630551683     )
losses.append(0.010327494110029721       )
accuracies.append(0.9956557458673571     )
losses.append(0.010327552968800462       )
accuracies.append(0.995500149372635      )
losses.append(0.010256901221178852       )
accuracies.append(0.9954503584943238     )
losses.append(0.010205911223799635       )
accuracies.append(0.9956993128858793     )
losses.append(0.010221393910727725       )
accuracies.append(0.9957304321848237     )
losses.append(0.010172773370120937       )
accuracies.append(0.9956557458673571     )
losses.append(0.01021242646574833        )
accuracies.append(0.9956993128858793     )
losses.append(0.009933313232175606       )
accuracies.append(0.9960042820155347     )
losses.append(0.009960871510519578       )
accuracies.append(0.9957615514837682     )
testAccuracies.append(0.8227195777733519 )

plt.figure()
plt.plot(losses, label='Loss')
plt.legend()
plt.show()
plt.figure()
plt.plot(accuracies, label='Training Set Accuracy')
plt.plot([0,10,20,30,40,50], testAccuracies, label='Test Set Accuracy')
plt.legend()
plt.show()