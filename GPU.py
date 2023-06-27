class User():
    def __init__(self, data):
        self.data = data
    def __str__(self):
        if len(self.data) == 0:
            return "\033[0;31;40m[]\033[0m"
        else:
            return self.data.__str__()
    def __len__(self):
        if len(self.data) == 0:
            return 2
        else:
            return self.data.__str__().__len__()


def check():
    def find_user(GPU_ID):
        user = set()
        for pid in GPU_dict[GPU_ID]:
            for i in USR_PID:
                if pid==i[1] or pid==i[2]:
                    user.add(i[0])
                    break
        return User(list(user)) if len(user) else User([])

    with open('GPU.txt') as f:
        L = f.readlines()
        G = L[0][:-1].split(' ')
        if len(G)==1 and G[0] == '':
            print('No running processes found ！！！！！！！！！！！！！')
            exit()
        U = L[1][:-1].split(' ')
        GPU_PID = [(x.split('*')[0], x.split('*')[1]) for x in G]
        USR_PID = [(x.split('*')[0], x.split('*')[1], x.split('*')[2]) for x in U]

        GPU_memery = list(map(lambda x: x[:-9], L[2][:-1].split('+')))
        GPU_util = list(map(lambda x: x.split(' ')[-1], GPU_memery))
        GPU_memery = list(map(lambda x: ''.join(x.split(' ')[:-1]), GPU_memery)) # join 拼接字符串
        GPU_dict = {}
        for gpu_id in range(7):
            GPU_dict.update({gpu_id:set()})
        for (i,pid) in GPU_PID:
            GPU_dict[int(i)].add(pid)
        print('-'*110)
        for gpu_id in range(7):
            usr = find_user(gpu_id)
            print('| GPU: {} user: '.format(gpu_id), usr, end='')
            print(GPU_memery[int(gpu_id)].rjust(88-len(usr)), end='')
            print(GPU_util[int(gpu_id)].rjust(5))
            # print('——'*56)
            # print('')
        print('-'*110)


if __name__ == "__main__":
    check()
    


