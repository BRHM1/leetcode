var deckRevealedIncreasing = function (deck) {
    const res = Array(deck.length)
    const queue = Array(deck.length)
    deck.sort((a, b) => a - b)
    for (let i = 0; i < queue.length; i++) queue[i] = i
    for (let card of deck) {
        let idx = queue.shift()
        res[idx] = card
        // skip the next index
        if (queue.length) queue.push(queue.shift())
    }
    return res
};
var removeKdigits = (num, k) => {
    let stack = []
    if (num.length === k) return "0"
    for (let n of num) {
        while (stack.at(-1) > n && k > 0) {
            stack.pop();
            k--
        }
        stack.push(n)
        while (stack.at(-1) === "0" && stack.length === 1) stack.pop()
    }
    return stack.slice(0, stack.length - k).join('') || "0"
}

var trap = function (height) {
    let water = 0, units = 0, diff = 0
    let stack = []
    let flag = false
    for (let i = 0; i < height.length; i++) {
        while(units){
            stack.push(stack.at(-1))
            units--
        }
        units = 0
        while (height[i] > stack.at(-1)) {
            flag = true
            if (stack.at(-2) < stack.at(-1) || stack.length === 1) break
            diff += stack.pop()
            units++
        }
        if(flag){
             water += Math.min( height[i], stack.at(-1)) * units - diff
            }
        stack.push(height[i])
        diff = 0
        flag = false
    }
    return water
};

// make a deep clone for object 
const foo = {
    firstname : "ahmed",
    lastname: {
        f : "ibrahiem",
        l : "saoud"
    }
}

const deepCopy = (obj) => {
    let res = {}
    for(let key in obj){
        if(typeof obj[key] === "object"){
            res[key] = deepCopy(obj[key])
        }else {
            res[key] = obj[key]
        }
    }
    return res
}

