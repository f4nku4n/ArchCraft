import logging
import sys
import time


class Log(object):
    _logger = None

    @classmethod
    def __get_logger(cls):
        if Log._logger is None:
            logger = logging.getLogger("EvoCNN")
            formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
            file_handler = logging.FileHandler("main.log")
            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.formatter = formatter
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
            Log._logger = logger
            return logger
        else:
            return Log._logger

    @classmethod
    def info(cls, _str):
        cls.__get_logger().info(_str)

    @classmethod
    def warn(cls, _str):
        cls.__get_logger().warn(_str)


class Utils:
    @classmethod
    def save_population_after_mutation(cls, _str, gen_no):
        file_name = './populations/mutation_%02d.txt' % (gen_no)
        with open(file_name, 'w') as f:
            f.write(_str)

    @classmethod
    def read_template(cls):
        _path = './template.py'
        part1 = []
        part2 = []

        f = open(_path, encoding='utf-8')
        f.readline()  # skip this comment
        line = f.readline().rstrip()
        while line.strip() != '#generated_code':
            part1.append(line)
            line = f.readline().rstrip()
        # print('\n'.join(part1))

        line = f.readline().rstrip()  # skip the comment '#generated_code'
        while line.strip() != '"""':
            part2.append(line)
            line = f.readline().rstrip()
        # print('\n'.join(part3))
        return part1, part2

    @classmethod
    def generate_pytorch_file(cls, indi):
        code = indi.code

        part1, part2 = cls.read_template()
        _str = []
        current_time = time.strftime("%Y-%m-%d  %H:%M:%S")
        _str.append('"""')
        _str.append(current_time)
        _str.append('"""')
        _str.extend(part1)
        _str.append('\ncode = %s' % str(code))
        _str.extend(part2)
        # print('\n'.join(_str))
        file_name = 'scripts/%s.py' % (indi.id)
        script_file_handler = open(file_name, 'w', encoding='utf-8')
        script_file_handler.write('\n'.join(_str))
        script_file_handler.flush()
        script_file_handler.close()
