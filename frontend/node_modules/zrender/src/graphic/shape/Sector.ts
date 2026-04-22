import Path, { PathProps } from '../Path';
import * as roundSectorHelper from '../helper/roundSector';

export class SectorShape {
    cx = 0
    cy = 0
    r0 = 0
    r = 0
    startAngle = 0
    endAngle = Math.PI * 2
    clockwise = true
    /**
     * Corner radius of sector
     *
     * clockwise, from inside to outside, four corners are
     * inner start -> inner end
     * outer start -> outer end
     *
     * 5               => [5, 5, 5, 5]
     * [5]             => [5, 5, 0, 0]
     * [5, 10]         => [5, 5, 10, 10]
     * [5, 10, 15]     => [5, 10, 15, 15]
     * [5, 10, 15, 20] => [5, 10, 15, 20]
     */
    cornerRadius: number | number[] = 0
}

export interface SectorProps extends PathProps {
    shape?: Partial<SectorShape>
}

class Sector extends Path<SectorProps> {

    shape: SectorShape

    constructor(opts?: SectorProps) {
        super(opts);
    }

    getDefaultShape() {
        return new SectorShape();
    }

    buildPath(ctx: CanvasRenderingContext2D, shape: SectorShape) {
        roundSectorHelper.buildPath(ctx, shape);
    }

    isZeroArea() {
        return this.shape.startAngle === this.shape.endAngle
            || this.shape.r === this.shape.r0;
    }
}

Sector.prototype.type = 'sector';

export default Sector;
