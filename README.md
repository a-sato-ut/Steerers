# Steerers

私はCVPR2024の論文"[Steerers: A framework for rotation equivariant keypoint descriptors](https://openaccess.thecvf.com/content/CVPR2024/papers/Bokman_Steerers_A_Framework_for_Rotation_Equivariant_Keypoint_Descriptors_CVPR_2024_paper.pdf)"の内容の実装を行なった．

このREADMEでは，まず，この論文の要旨についてまとめる．
次に，なぜこの論文が重要であるのかを「この論文の技術的な核の部分」と「なぜこの論文がトップカンファレンスにされたと考えられるか」の二つの観点から記述する．
最後に，私が実装を行なった部分と，コードの実行の仕方について説明する．

## Abstract

この論文では，回転に頑健なkeypoint記述の，新しいフレームワークを提案している．
近年は，機械学習，特に深層学習を用いたkeypoint記述が，その精度の高さからよく用いられている．
しかし，これらの機械学習を用いたkeypoint記述では回転に対する**不変性**が一般には保証されていない．
そのため，学習やモデルを工夫することで回転に**不変**なkeypoint記述を目指すことが提案されてきたが，これらの手法は回転がほとんどない場合の精度を少し犠牲にしている．
そこで，この論文では回転がほとんどない場合の精度を全く犠牲にすることなく，回転に**等価**なkeypoint記述を達成するフレームワークを提案している．
具体的には，「画像の回転」を「記述空間」で再現する行列を求めることで，keypoint記述を行うモデルやそのパラメータは固定したまま，軽量な計算で回転した画像のkeypoint記述を得ることを可能にする．

## Why this paper is important.

### What the technical core is.
この論文の技術的な核の部分は，「記述空間における線形変換を学習することで，入力画像の回転をエンコードする」というアプローチにある．
この線形変換「Steerer」は，画像が回転されたかのように記述を変換することができるため，回転した画像を再度ネットワークで処理することなく，既存の記述を回転対応に調整することが可能となる．
このフレームワークにより，既存のディスクリプタを使用しながら，軽量な計算で回転に等価なkeypoint記述を得ることができる．
既存のkeypoint記述をそのまま使うことができ，テスト時の軽量なオーバーヘッドの追加によって回転に頑健なkeypointマッチングを行うことができるこのフレームワークは革新的であるといえる．

### Why the paper is accepted.
この論文が採択された大きい理由の一つが，その適用範囲の広さにあると考えられる．

既存のほぼ任意のkeypoint記述に対して適用することができる手法であり，テスト時のオーバーヘッドも軽量であるため，この手法を適用することのデメリットは非常に小さい．
一方で，keypointマッチングの実験においてはSoTAの性能を得ることができ，この手法を適用するメリットは非常に大きい．
したがって，この手法は広範なシナリオにおいて有用であると考えられる．

また，この論文は，ある仮定を満たすようなkeypoint記述について，厳密なSteererが存在すること，すなわち，ある線形変換が存在して，画像の回転を記述空間上で再現できることを証明している．
この仮定は，回転についてほとんど考慮せずに訓練されたkeypoint記述についてもたしかに成り立つと考えられるような尤もらしいものであり，理論的な観点からも，適用可能範囲が広い貢献を与えていると言える．

このような，技術的/理論的に適用範囲の広い手法/定理を与えていることが，この論文の貢献を大きくし，トップカンファレンスへの採択につながっていると考えられる．

## What you have implemented
[Steererモデル](./Steerers/steerers.py)と，[Steererを用いて/用いないで2枚の画像のkeypointマッチングを行うPythonコード](./Steerers/matcher.py)，および[それらを呼び出して結果を保存するコード](./main.py)の実装を行なった．
Steererモデルは1つの行列であるので，1つの全結合層のみからなるモデルとして実装した．
Steererを用いて画像のkeypointマッチングを行う際には，一方の画像のkeypoint記述を90度ずつ回転させ，それぞれの場合でマッチングを行い，最もマッチしたkeypoint数が多い回転を採用することで，高速に回転を考慮したkeypointマッチングを行う．

以下の流れで[main.py](./main.py)を実行することで，[images/im_A.jpg](./images/im_A.jpg)と[images/im_A_rot.jpg](./images/im_A_rot.jpg)を，Steererを用いないでkeypointマッチングを行なった場合（[images/without_steerer_result.png](./images/without_steerer_result.png)）と，Steererを用いてkeypointマッチングを行なった場合（[images/with_steerer_result.png](./images/with_steerer_result.png)）の結果を得ることができる．

### Setup
```
./setup.sh
```

### Run main.py
```
python main.py --im1 images/im_A.jpg --im2 images/im_A_rot.jpg --wo images/without_steerer_result.png --w images/with_steerer_result.png
```
