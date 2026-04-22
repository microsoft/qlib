import Gradient, { GradientColorStop, GradientObject } from './Gradient';
export interface RadialGradientObject extends GradientObject {
    type: 'radial';
    x: number;
    y: number;
    r: number;
}
declare class RadialGradient extends Gradient {
    type: 'radial';
    x: number;
    y: number;
    r: number;
    constructor(x: number, y: number, r: number, colorStops?: GradientColorStop[], globalCoord?: boolean);
}
export default RadialGradient;
