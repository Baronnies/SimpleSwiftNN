

// Main Lib File
// Remark: This library is a simple Swift Library

import Foundation

// Matrix Implementation
// Remark: This is not an effective implementation, rather a simple one, made to facilitate the understanding of matrix operations

class Matrix {
    let rows: Int
    let cols: Int
    
    var data: [[Double]] = []
    
    // Returning the absolute value of a matrix
    var absValue: Matrix {
        let resultingMatrix = Matrix(rows: rows, cols: cols)
        
        for i in 0...(rows-1) {
            for j in 0...(cols-1) {
                    resultingMatrix.data[i][j] = abs(data[i][j])
                }
            }
        return resultingMatrix
    }
    
    init(rows: Int, cols: Int) {
        self.rows = rows
        self.cols = cols
        
        
        for i in 0...(rows-1) {
            data.append([])
            
            for _ in 0...(cols-1) {
                data[i].append(1)
            }
        }
        
    }
   
    
    // Adding a constant to each element of the matrix
    func add(const: Double) {
        for i in 0...(rows-1) {
            for j in 0...(cols-1) {
                data[i][j] += const
            }
        }
    }
    
    // Adding two matrices together
    func addElementwise(m: Matrix) {
        for i in 0...(m.rows-1) {
            for j in 0...(m.cols-1) {
                data[i][j] += m.data[i][j]
            }
        }
    }
    
    /**********************************************************************/
    // Most Important operation: Dot Product of 2 matrices
    
    static func dotProduct(a: Matrix, b: Matrix) -> Matrix {
        guard a.cols == b.rows else {
            print("Dot Product Failed")
            return Matrix(rows: 1, cols: 1)
            
        }
        
        let resultingMatrix = Matrix(rows: a.rows, cols: b.cols)
        
        for i in 0..<resultingMatrix.rows {
            for j in 0..<resultingMatrix.cols {
                
                var sum: Double = 0
                
                for k in 0..<a.cols {
                    sum += a.data[i][k] * b.data[k][j]
                    
                }
                
                resultingMatrix.data[i][j] = sum
            }
        }
        
        return resultingMatrix
    }
    
    // The class allows us to take array inputs and convert them into Matrix objects
    static func fromArray(arr: [Double]) -> Matrix {
        let result = Matrix(rows: arr.count, cols: 1)
        
        for index in 0..<arr.count {
            result.data[index][0] = arr[index]
        }
        
        return result
    }
    
    // Multyplying together the elements of two matrices
    func hadamardProduct(m: Matrix) {
        for i in 0...(rows-1) {
            for j in 0...(cols-1) {
                data[i][j] *= m.data[i][j]
            }
        }
    }
    
    // Any Function passed as Parameter will be appied to each element of the matrix
    // Used for the activation function
    func map(fn: (Double) -> Double) {
        for i in 0..<self.rows {
            for j in 0..<self.cols {
                self.data[i][j] = fn(self.data[i][j])
            }
        }
    }
    
    static func map(m: Matrix, fn: (Double) -> Double) -> Matrix {
        let resultingMatrix = Matrix(rows: m.rows, cols: m.cols)
        
        for i in 0..<m.rows {
            for j in 0..<m.cols {
                resultingMatrix.data[i][j] = fn(m.data[i][j])
            }
        }
        
        return resultingMatrix
    }
    
    // Logging the Matrix
    func printMatrix() {
        for i in 0..<self.rows {
            var stringData = ""
            for j in 0..<self.cols {
                stringData.append(String(self.data[i][j]))
                stringData += " "
            }
            
            print(stringData)
        }
        print()
    }
    
    // Initializes the matrix with random values ranging from m to n
    func randomize(m: Double, n: Double) {
        for i in 0...(rows-1) {
            for j in 0...(cols-1) {
                data[i][j] = Double.random(in: m...n)
            }
        }
    }
    
    // Scalar -> Multyplying by a constant
    func scale(scalar: Double) {
        for i in 0...(rows-1) {
            for j in 0...(cols-1) {
                data[i][j] *= scalar
            }
        }
    }
    
    // Substraction of matrices
    static func substractElementwise(_ m: Matrix, _ n: Matrix) -> Matrix {
        
        guard (m.rows == n.rows && m.cols == n.cols) else {
            return Matrix(rows: 0, cols: 0)
        }
        
        let resultingMatrix = Matrix(rows: m.rows, cols: m.cols)
        
        for i in 0...(m.rows-1) {
            for j in 0...(m.cols-1) {
                resultingMatrix.data[i][j] = m.data[i][j] - n.data[i][j]
            }
        }
        
        return resultingMatrix
    }
    
    // The class allows us to take matrix inputs and convert them into an Array
    static func toArray(m: Matrix) -> [Double] {
        var resultingArray: [Double] = []
        
        for i in 0...(m.rows-1) {
            for j in 0...(m.cols-1) {
                resultingArray.append(m.data[i][j])
            }
        }
        
        
        return resultingArray
    }
    
    // Transpose the Matrix
    /*  Example :
     
     [ a b c ]   to   [ a d ]
     [ d e f ]        [ b e ]
                      [ c f ]
     */
    
    // Transpose the Object itself
    func transpose() {
        
        for i in 0..<self.rows {
            for j in 0..<self.cols {
                self.data[j][i] = self.data[i][j]
            }
        }
        
    }
    
    // Static method: Return the transposed matrix
    static func transpose(m: Matrix)-> Matrix {
        let resultingMatrix = Matrix(rows: m.cols, cols: m.rows)
        
        for i in 0..<m.rows {
            for j in 0..<m.cols {
                resultingMatrix.data[j][i] = m.data[i][j]
            }
        }
        
        return resultingMatrix
    }
    
    
// Activation function

    //  The activation function used for the neural network will be the sigmoid function
    static func sigmoid(z: Double) -> Double {
        return 1.0 / (1.0 + exp(-z))
    }
    
    
    // Partially Derivated Sigmoid: Outputs of the feedforward are ouputs of sigmoid already
    static func dsigmoid(z: Double) -> Double {
        return (1 - sigmoid(z: z))
    }
    
}


// Neural Network Class and Main Part
class NeuralNetwork {
    
    // Weights and Biases of the neural network
    var weights: [Matrix] = []
    var biases:  [Matrix] = []
    
    // The feedforward output of each layer
    var layerResult: [Matrix] = []
    
    // Learning Rate of the NN: Change at your own risk
    static let learningRate: Double = 0.01
    
    // Percentage of training dataset / total dataset: Change at your own risk
    static let trainingSize: Double = 0.95
    
    // Minimal dataset size required: Change at your own risk
    static let minimalDatasetSize: Int = 1000
    
    // Numbers of layers in the NN, used to speed up calculations
    var nnLayers: Int = 0
    
    // Dataset Used to Train the NN
    var trainingDataSet: [Data] = []
    
    // Dataset Used to Test the NN
    var testingDataSet: [Data] = []
    
    // Int... means that several hidden layers can be entered like so:
    // NeuralNetwork(6,3,4,5, numO: 6, data: data)
    init(_ numI: Int, _ numH: Int..., numO: Int, data: [Data]) {
        
        // Initializing the Datasets
        initData(data: data)

        // Randomizing all layers and setting Up the NN
        initNN(numI, numH, numO)
    }
    
    
    // Function to Initialize the datasets
    func initData(data: [Data]) {

        // Randomizing the dataset
        var randomData = data.shuffled()
        let trainingElements = Int(NeuralNetwork.trainingSize * Double(data.count) )
        
        // Separating the known Data into 2 categories:
        
        // Usually 60% of the data = Training data
        self.trainingDataSet = randomData.dropLast(data.count - trainingElements)
        
        // Usually 40% of the data = Testing data
        stride(from: trainingElements, to: randomData.count, by: 1).map {
            testingDataSet.append(randomData[$0])
        }
        
    }
    
    
    // Function to Initialize the neural network
    func initNN(_ numI: Int, _ numH: [Int], _ numO: Int)  {
        
        // Init the neural network with input, hidden and output neurons
        var layers: [Int] = numH
        layers.insert(numI, at: 0)
        layers.append(numO)
        
        // Number of weight layers, not neuron layers
        nnLayers = layers.count - 1
        
        // Init the weigths and biases
        for index in 1..<(layers.count) {
            
            let layerMatrix = Matrix(rows: layers[index], cols: layers[index - 1])
            layerMatrix.randomize(m: 0, n: 1)
            
            self.weights.append(layerMatrix)
            
            // The bias Matrix is in fact a vector
            let biasMatrix = Matrix(rows: layers[index], cols: 1)
            biasMatrix.randomize(m: 0, n: 1)
            
            self.biases.append(biasMatrix)
        }
        
        
    }
    
    
    // Feedforward implementation
    func feedforward(input: [Double]) -> Matrix {
        
        // Preparing to store the result of each layer's Process
        layerResult = []
        
        // Transforming the array into a matrix
        let inputMatrix = Matrix.fromArray(arr: input)
        layerResult.append(inputMatrix)
        
        // The output of each layer is stored in currentOutput
        // Note: The input Layer does not have a bias, because it would be overriden by the input
        
        var currentOutput: Matrix = Matrix.dotProduct(a: self.weights[0], b: inputMatrix)
        currentOutput.addElementwise(m: self.biases[0])
        currentOutput.map(fn: Matrix.sigmoid)
        
        layerResult.append(currentOutput)
        
        // Iterate through all the layers to obtain the output
        for index in 1..<nnLayers {

            currentOutput = Matrix.dotProduct(a: self.weights[index], b: currentOutput)

            // We then add the bias to each neuron in the layer
            currentOutput.addElementwise(m: self.biases[index])

            // The Output of each neuron in the layer has to be a Double value between 0 and 1
            currentOutput.map(fn: Matrix.sigmoid)

            layerResult.append(currentOutput)
        }
        
        
        return currentOutput
    }
    
    
    // Stochastic Gradient Descent
    func backpropagation(matrixOutputs: Matrix, arrTarget: [Double]) {
        
        // Convert the target to a Matrix
        let matrixTargets = Matrix.fromArray(arr: arrTarget)
        
        // Calculate the error of the current iteration
        // Error = Target - Guess
        let outputError = Matrix.substractElementwise(matrixTargets, matrixOutputs)
        var precedentLayerError = outputError
        
        
        // Backpropagate through the reversed layers
        for index in 0..<nnLayers {
            
            // Calculate the Gradients
            // DSigmoid is the "derivated" version of sigmoid
            let gradients = Matrix.map(m: layerResult[nnLayers - index], fn: Matrix.dsigmoid)
            
            gradients.hadamardProduct(m: precedentLayerError)
            // Rate at which the NN learns
            gradients.scale(scalar: NeuralNetwork.learningRate)
            
            
            // Calculate Deltas
            let weightTransposed = Matrix.transpose(m: layerResult[nnLayers - index - 1])
            let weightDeltas = Matrix.dotProduct(a: precedentLayerError, b: weightTransposed)
           
            
            // Adjust the weight by deltas
            weights[nnLayers - index - 1].addElementwise(m: weightDeltas)
            
            // Adjust the bias by gradients
            biases[nnLayers - index - 1].addElementwise(m: gradients)
            
            // Calculate the layer's Error
            let transposedWeight = Matrix.transpose(m: weights[nnLayers - index - 1])
            let weightLayerErrors = Matrix.dotProduct(a: transposedWeight, b: precedentLayerError)
            
            precedentLayerError = weightLayerErrors
        }
    }
    
    
    // Training the neural Network
    func train() {
        print("Starting the training process...")
        
        let trainingSize = trainingDataSet.count
        let testingSize = testingDataSet.count
        
        // Iterate over the training dataset
        for index in 0..<trainingSize {
            let trainingData = trainingDataSet[index]
            
            // Trains the network by feedforwarding the training data,
            // Comparing it to the target and adjusting the weights/biases
            let nnOutput = feedforward(input: trainingData.input)
            backpropagation(matrixOutputs: nnOutput, arrTarget: trainingData.target)
        }
        
        print("Finished the training process with \(trainingSize) iterations")
        print()
       
        // Testing the quality of the NN by averaging its accuracy
        let averageAccuracy: Matrix = Matrix(rows: testingDataSet[0].target.count, cols: 1)
        
        for index in 0..<testingSize {
            
            let currentTestingData = testingDataSet[index]
            
            // Computing the Error of the NN
            let nnOutput = feedforward(input: currentTestingData.input)
            let matrixTarget = Matrix.fromArray(arr: currentTestingData.target)
            
            // This iteration's Error is added to the average
            let iterationError = Matrix.substractElementwise(matrixTarget, nnOutput)
            averageAccuracy.addElementwise(m: iterationError.absValue)
        }
        
        averageAccuracy.scale(scalar: 1/Double(testingSize))
        
        
        // Converting it into a percentage
        var finalAverage: Double = Double(Matrix.toArray(m: averageAccuracy).reduce(0, +))
        finalAverage *= (100/Double(averageAccuracy.rows))
        finalAverage = 100 - abs(finalAverage)
        
        print("Average accuracy of the neural network: \(finalAverage)")

    }
    
}


//  Class that will generate the labeled training data
class Data {
    
    // The input data
    let input:  [Double]
    
    // The label of the data (or target)
    let target: [Double]
    
    init(_ input: [Double], _ target: [Double]) {
        self.input = input
        self.target = target
    }
    
    // Converts quantity of raw data into Data instances
    static func fromArrays(_ inputs: [[Double]], _ targets: [[Double]]) -> [Data] {
        
        let inputSize: Int = inputs.count
        var resultingData: [Data] = []
        
        guard inputSize == targets.count else {
            return [Data([0], [0])]
        }
        
        for index in 0..<inputSize {
            resultingData.append(Data(inputs[index], targets[index]))
        }
        
        return resultingData
    }
    
    // "Makes" more data by repeating elements (not really a good option, used just to test)
    static func createData(_ finalSize: Int, _ data: [Data]) -> [Data] {
        var resultingData: [Data] = []
        
        for i in 0..<finalSize {
            let newData = data[i%(data.count - 1)]
            resultingData.append(newData)
        }
        
        return resultingData
    }
    
    
}


// File for running the NNs

// The XOR Problem that the NN has to solve
let targets: [[Double]] = [[0], [0], [1], [1]]
let inputs: [[Double]] = [[2], [4], [1], [3]]

// Convert the raw data into data instances
var data = Data.fromArrays(inputs, targets)
data = Data.createData(500, data)

// Initialize the NN and trains it
let newNN = NeuralNetwork(1, 5, 3, numO: 1, data: data)
newNN.train()

// Play with this toy Swift Library now
// Add more neuron layers, modify the learning rate, the minimalSize
// Or invent new problems for the NN to solve

