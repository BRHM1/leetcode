var deckRevealedIncreasing = function(deck) {
    const n = deck.length;
    deck.sort((a, b) => a - b);
    const revealed = [];
    revealed.unshift(deck[n - 1]);
    for (let i = n - 2; i >= 0; i--) {
        revealed.unshift(revealed.pop());
        revealed.unshift(deck[i]);
        // console.log(revealed)
    }
    return revealed;
};
console.log(deckRevealedIncreasing([17, 13, 11, 2, 3, 5, 7]))
// console.log(deckRevealedIncreasing([1,2,3,4,5]))
// console.log(deckRevealedIncreasing([17, 13, 11, 2, 3, 5,7,19]))
// console.log(deckRevealedIncreasing([1 , 1000]))
// console.log(deckRevealedIncreasing([1, 2, 3, 4]))