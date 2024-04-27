 const findShortestPath = (col , target) => {
        let searchSpace = ring + ring
        let shortest = Infinity
        for(let i = 0; i < searchSpace.length; i++){
            if(searchSpace[i] === target) {
                shortest = Math.min(shortest, Math.abs(i - col ));
            }
        }
        return shortest;
    }