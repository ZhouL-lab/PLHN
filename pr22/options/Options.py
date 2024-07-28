from options.BasicOptions import BaseOptions


class Options_x(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--name', type=str, default='Tumor_seg', help='name_of_the_project')
        self.isTrain = True
        return parser
