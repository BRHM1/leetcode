for(let i = 0; i < newgrid[0].length; i++){
        let colwindow = []
        for(let j = 0; j < newgrid.length; j++){
            colwindow.push(newgrid[j][i])
            if(colwindow.length === 3){

                colwindow.shift()
            }
        }
    }