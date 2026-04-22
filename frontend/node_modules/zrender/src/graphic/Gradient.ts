// TODO Should GradientObject been LinearGradientObject | RadialGradientObject
export interface GradientObject {

    id?: number

    type: string

    colorStops: GradientColorStop[]

    global?: boolean
}

export interface InnerGradientObject extends GradientObject {
    __canvasGradient: CanvasGradient
    __width: number
    __height: number
}

export interface GradientColorStop {
    offset: number
    color: string
}

export default class Gradient {

    id?: number

    type: string

    colorStops: GradientColorStop[]

    global: boolean

    constructor(colorStops: GradientColorStop[]) {
        this.colorStops = colorStops || [];
    }

    addColorStop(offset: number, color: string) {
        this.colorStops.push({
            offset,
            color
        });
    }
}