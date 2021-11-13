author: Li Qi
summary: A demo implimentation of zk-snarks based on Pinocchio protocol
id: pinocchio-protocol
tags: zeroknowledgeproof, zk-snarks, pinocchio, golang
categories: zeroknowledgeproof
status: Published
feedback link: https://github.com/InspiringGroupCodelabs/InspiringGroupCodelabs.github.io/issues

# Pinocchio-based zksnarks

## **Codelab Overview**

Duration: 0:01:00

接下来，本文档将介绍零知识证明和zksnarks，并提供一个简单的基于匹诺曹协议（ Pinocchio protocol）的实现。

Positive
: 如需完整的代码项目请访问 https://github.com/liqi16/pinocchio-protocol-zksnarks.git

Positive
: 如需学习完整的匹诺曹协议的请访问 https://inspiringgroup-codelab.readthedocs.io/zh_CN/latest/zkp/pinocchio.html

## 零知识证明

Duration: 0:01:00

### 为什么需要可证明的计算？

随着技术的发展，计算能力表现出不对称的特性，例如云计算等拥有大量的算力，而移动设备等算力十分有限。因此，一些计算能力较弱的客户端设备希望通过外包计算的方式，将计算任务外包给算力强大的设备。而此时，这些客户端就希望在得到计算结果的同时，可以验证结果的正确性，以防止偶然的错误或恶意的攻击。同时，从另一方面，提供外包计算服务的服务商也希望可以证明自己的工作，这样，他们既可以要求更高的价格，又可以摆脱不必要的责任。

### 什么是零知识证明?

零知识证明是指证明者能够在不向验证者提供任何有用的信息的情况下，使验证者相信某个论断是正确的。

一个简单的例子：A要向B证明自己拥有某个房间的钥匙（通常情况下，我们把A称为证明者Prover，B称为验证者Verifier），假设该房间只能用钥匙打开锁，其他任何方法都打不开，而且B确定该房间内有某一物体。此时A用自己拥有的钥匙打开该房间的门，然后把物体拿出来出示给B，从而证明自己确实拥有该房间的钥匙。**在这个过程中，证明者A能够在不给验证者B看到钥匙的情况下，使B相信他是有钥匙开门的。**

我们可以通过零知识证明的思路，实现可证明的计算。

## zksnarks

Duration: 0:01:00

zk-SNARK（ Zero-Knowledge Succinct Non-Interactive Argument of Knowledge）是零知识证明的一种形式，它只适用于满足QAP（[Quadratic Arithmetic Program](https://link.springer.com/content/pdf/10.1007/978-3-642-38348-9_37.pdf)）的定义形式的计算问题。zksnarks具有以下性质：

简明 (Succinctly) : 独立于计算量，证明是恒定的，小尺寸的。

非交互性 (Non-interactive) : 证明只要一经计算就可以在不直接与 prover 交互的前提下使任意数量的 verifier 确信。

可论证的知识 (Argument of Knowledge) :对于陈述是正确的这点有不可忽略的概率，即无法构造假证据；并且 prover 知道正确陈述的对应值（即：证据）。

零知识( zero-knowledge) : 很难从证明中提取任何知识，即它与随机数无法区分。

匹诺曹协议是一种简单的实现zksnarks协议：（[Pinocchio protocol](https://eprint.iacr.org/2013/279.pdf)）。

## 代码实现框架

Duration: 0:01:00

匹诺曹协议的实现方法参考 [Go-snark](https://github.com/shamatar/go-snarks.git)和[go-snark-study](https://github.com/arnaucube/go-snark-study)。这里使用[V神(Vitalik Buterin)的例子](https://medium.com/@VitalikButerin/zk-snarks-under-the-hood-b33151a013f6)进行实现。完整代码见[GitHub](https://github.com/liqi16/pinocchio-protocol-zksnarks.git)。代码运行方式：

```shell
go get github.com/arnaucube/go-snark
go get github.com/arnaucube/go-snark/circuitcompiler
go run main.go
```

实现的总体架构如下：
1. verifier初始化
2. prover提供证明
3. verifier验证证明

```go
func main() {

	//verifier初始化
	flatCode := PrepareCircuit()

	circuit := CompileCircuit(flatCode)

	setup := TrustedSetup(circuit)

	pk := setup.Pk
	vk := setup.Vk
  
  /*verfier将circuit,pk交给prover*/

	//prover提供证明
	inputs := PrepareInputAndOutput()

	proof := GenerateProofs(circuit, pk, inputs)
  
  /*prover将proof,inputs.Public[35]交给prover*/

	//verifier验证证明
	verified := VerifyProofs(vk, inputs.Public, proof)

	if !verified {
		fmt.Println("proofs not verified")
	} else {
		fmt.Println("Proofs verified")
	}

}
```

## 准备电路 - PrepareCircuit

Duration: 0:01:00

我们用到的函数是y=x^3 + x + 5。将这个函数拍平，转换为“一个等式中最多含有一次乘法的形式”。这样我们就得到了一个拍平的函数。

```go
func PrepareCircuit() string {

	flatCode := `
	func exp3(private a):
		b = a * a
		c = a * b
		return c

	func main(private s0, public s1):
		s3 = exp3(s0)
		s4 = s3 + s0
		s5 = s4 + 5
		equals(s1, s5)
		out = 1 * 1
	`
	return flatCode
}
```

## 编译电路 - CompileCircuit

Duration: 0:01:00

我们将电路编译，并转换为R1CS。

```go
func CompileCircuit(flatCode string) circuitcompiler.Circuit {
	// parse the code
	parser := circuitcompiler.NewParser(strings.NewReader(flatCode))
	circuit, err := parser.Parse()
	panicErr(err)
	fmt.Println("circuit", circuit)

	a, b, c := circuit.GenerateR1CS()
	fmt.Println("\nR1CS:")
	fmt.Println("circuit.R1CS.A", a)
	fmt.Println("circuit.R1CS.B", b)
	fmt.Println("circuit.R1CS.C", c)

	return *circuit

}
```

我们可以看到如下的输出：

```
R1CS:
circuit.R1CS.A [[0 0 1 0 0 0 0 0] [0 0 1 0 0 0 0 0] [0 0 1 0 1 0 0 0] [5 0 0 0 0 1 0 0] [0 0 0 0 0 0 1 0] [0 1 0 0 0 0 0 0] [1 0 0 0 0 0 0 0]]
circuit.R1CS.B [[0 0 1 0 0 0 0 0] [0 0 0 1 0 0 0 0] [1 0 0 0 0 0 0 0] [1 0 0 0 0 0 0 0] [1 0 0 0 0 0 0 0] [1 0 0 0 0 0 0 0] [1 0 0 0 0 0 0 0]]
circuit.R1CS.C [[0 0 0 1 0 0 0 0] [0 0 0 0 1 0 0 0] [0 0 0 0 0 1 0 0] [0 0 0 0 0 0 1 0] [0 1 0 0 0 0 0 0] [0 0 0 0 0 0 1 0] [0 0 0 0 0 0 0 1]]
```

## 初始化 - TrustedSetup

Duration: 0:01:00

根据函数生成公开密钥PK和验证密钥VK。

```go
func TrustedSetup(circuit circuitcompiler.Circuit) snark.Setup {

	// R1CS to QAP
	alphas, betas, gammas, _ := snark.Utils.PF.R1CSToQAP(circuit.R1CS.A, circuit.R1CS.B, circuit.R1CS.C)
	fmt.Println("QAP")
	fmt.Println(alphas)
	fmt.Println(betas)
	fmt.Println(gammas)

	// calculate trusted setup
	setup, err := snark.GenerateTrustedSetup(len(circuit.Signals), circuit, alphas, betas, gammas)
	panicErr(err)
	fmt.Println("\nt:", setup.Toxic.T)//私钥，可销毁

	// remove setup.Toxic
	var tsetup snark.Setup
	tsetup.Pk = setup.Pk
	tsetup.Vk = setup.Vk

	return tsetup
}
```

## 准备输入输出 - PrepareInputAndOutput

Duration: 0:01:00

输入x=3，按照函数y=x^3 + x + 5，输出值为y=35。

```go
func PrepareInputAndOutput() circuitcompiler.Inputs {

	input := `[
		3
	]
	`

	output := `[
		35
	]
	`

	var inputs circuitcompiler.Inputs
	err := json.Unmarshal([]byte(input), &inputs.Private)
	panicErr(err)
	err = json.Unmarshal([]byte(output), &inputs.Public)
	panicErr(err)

	return inputs

}
```

## 生成证明 - GenerateProof

Duration: 0:10:00

prover通过函数的输入输出，以及verifier的初始化参数，生成证明。

```go
func GenerateProofs(circuit circuitcompiler.Circuit, pk snark.Pk, inputs circuitcompiler.Inputs) snark.Proof {

	// calculate wittness
	witness, err := circuit.CalculateWitness(inputs.Private, inputs.Public)
	panicErr(err)
	fmt.Println("\nwitness", witness)

	// flat code to R1CS
	a := circuit.R1CS.A
	b := circuit.R1CS.B
	c := circuit.R1CS.C
	// R1CS to QAP
	alphas, betas, gammas, _ := snark.Utils.PF.R1CSToQAP(a, b, c)
	_, _, _, px := snark.Utils.PF.CombinePolynomials(witness, alphas, betas, gammas)
	hx := snark.Utils.PF.DivisorPolynomial(px, pk.Z)

	fmt.Println(circuit)
	fmt.Println(pk.G1T)
	fmt.Println(hx)
	fmt.Println(witness)
	proof, err := snark.GenerateProofs(circuit, pk, witness, px)
	panicErr(err)

	fmt.Println("\n proofs:")
	fmt.Println(proof)

	return proof
}
```

## 验证结果 - VerifyProofs

```go
func VerifyProofs(vk snark.Vk, publicinputs []*big.Int, proof snark.Proof) bool {
	verified := snark.VerifyProof(vk, proof, publicinputs, true)
	return verified
}
```

我们可以看到如下输出：

```
✓ e(piA, Va) == e(piA', g2), valid knowledge commitment for A
✓ e(Vb, piB) == e(piB', g2), valid knowledge commitment for B
✓ e(piC, Vc) == e(piC', g2), valid knowledge commitment for C
✓ e(Vkx+piA, piB) == e(piH, Vkz) * e(piC, g2), QAP disibility checked
✓ e(Vkx+piA+piC, g2KbetaKgamma) * e(g1KbetaKgamma, piB) == e(piK, g2Kgamma)
```