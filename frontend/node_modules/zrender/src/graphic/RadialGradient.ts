import Gradient, {GradientColorStop, GradientObject} from './Gradient';

export interface RadialGradientObject extends GradientObject {
    type: 'radial'

    x: number
    y: number
    r: number
}
/**
 * x, y, r are all percent from 0 to 1 when globalCoord is false
 */
class RadialGradient extends Gradient {

    type: 'radial'

    x: number
    y: number
    r: number

    constructor(
        x: number, y: number, r: number,
        colorStops?: GradientColorStop[], globalCoord?: boolean
    ) {
        super(colorStops);
        // Should do nothing more in this constructor. Because gradient can be
        // declard by `color: {type: 'radial', colorStops: ...}`, where
        // this constructor will not be called.
        this.x = x == null ? 0.5 : x;

        this.y = y == null ? 0.5 : y;

        this.r = r == null ? 0.5 : r;

        // Can be cloned
        this.type = 'radial';

        // If use global coord
        this.global = globalCoord || false;
    }
}

export default RadialGradient;