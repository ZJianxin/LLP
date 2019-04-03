def train(model, device, train_loader, optimizer, epoch, criterion, log_interval=10):
    model.train()
    for batch_idx, (data, target, alpha) in enumerate(train_loader):
        data, target, alpha = data.to(device), target.to(device), alpha.to(device)
        alpha.requires_grad = False
        optimizer.zero_grad()
        output = model(data)
        #loss = BCELoss(weight=weight)(output, target)
        loss = criterion(output, target, alpha)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            alpha = torch.ones(target.size())/2.0
            alpha = alpha.to(device)
            test_loss += criterion(output, target, alpha).item()
            #pred = torch.where(output>0.5, 1, 0)
            pred = output > 0.5
            #correct += pred.eq(target.view_as(pred)).sum().item()
            correct += pred.float().eq(target).sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.8f}, accuracy: {}/{} ({:.0f}%)\n'.format(
          test_loss, correct, len(test_loader.dataset), 
          100.*correct/len(test_loader.dataset)))