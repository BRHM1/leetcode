var deckRevealedIncreasing = function(deck) {
    const res = Array(deck.length)
    const queue = Array(deck.length)
    deck.sort((a , b) => a - b)
    for(let i = 0; i < queue.length; i++) queue[i] = i
    for(let card of deck){
        let idx = queue.shift()
        res[idx] = card
        // skip the next index
        if(queue.length) queue.push(queue.shift())
    }
    return res
};
