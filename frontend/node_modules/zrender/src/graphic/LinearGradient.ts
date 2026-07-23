import Gradient, {GradientObject, GradientColorStop} from './Gradient';

export interface LinearGradientObject extends GradientObject {
    type: 'linear'

    x: number
    y: number
    x2: number
    y2: number
}
/**
 * x, y, x2, y2 are all percent from 0 to 1 when globalCoord is false
 */

export default class LinearGradient extends Gradient {

    type: 'linear'

    x: number
    y: number
    x2: number
    y2: number

    constructor(
        x: number, y: number, x2: number, y2: number,
        colorStops?: GradientColorStop[], globalCoord?: boolean
    ) {

        super(colorStops);

        // Should do nothing more in this constructor. Because gradient can be
        // declard by `color: {type: 'linear', colorStops: ...}`, where
        // this constructor will not be called.

        this.x = x == null ? 0 : x;

        this.y = y == null ? 0 : y;

        this.x2 = x2 == null ? 1 : x2;

        this.y2 = y2 == null ? 0 : y2;

        // Can be cloned
        this.type = 'linear';

        // If use global coord
        this.global = globalCoord || false;
    }
};
